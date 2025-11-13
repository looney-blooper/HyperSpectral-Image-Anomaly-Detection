# src/block_search_tf.py
import tensorflow as tf
from .block_tf import BlockEmbeddingTF , BlockFoldTF  # your extractor (channels-last)

class BlockSearchTF:
    """
    Compute match_vec for GT-HAD using folded reconstructions.
    Works with BlockFoldTF that assumes no-overlap (sh==ph, sw==pw).
    """

    def __init__(self, block_embedding: BlockEmbeddingTF, block_query):
        """
        block_embedding: instance of BlockEmbeddingTF used to produce block_query originally.
        block_query: tf.Tensor of shape [num_blocks, L] or [num_blocks, ph, pw, C]
        """
        self.block_embedding = block_embedding
        bq = tf.convert_to_tensor(block_query)
        if len(bq.shape) == 4:
            self.block_query = tf.reshape(bq, (tf.shape(bq)[0], -1))  # [num_blocks, L]
        else:
            self.block_query = bq
        self.num_blocks = int(self.block_query.shape[0])

    def compute_match_vec_from_batched_search_matrix(self, search_matrix_batched, info, orig_H=None, orig_W=None):
        """
        search_matrix_batched: [B, N, ph, pw, C] or [N, ph, pw, C] (if [N,...], B inferred as 1)
        info: dict returned by BlockEmbeddingTF.extract (new_h,new_w,ph,pw,sh,sw,padding)
        Returns match_vec: tf.Tensor of shape [num_blocks] with 0/1 floats
        """
        # normalize inputs
        sm = tf.convert_to_tensor(search_matrix_batched)
        if len(sm.shape) == 4:
            # [N, ph, pw, C] -> add batch dim
            sm = tf.expand_dims(sm, axis=0)  # [1, N, ph, pw, C]
        B = tf.shape(sm)[0]
        N = int(info['new_h']) * int(info['new_w'])

        # fold to image using your BlockFoldTF
        bf = BlockFoldTF()
        # bf.fold expects shape [B, N, ph, pw, C] and returns [B, H, W, C] (optionally cropped)
        search_back = bf.fold(sm, info, orig_H=orig_H, orig_W=orig_W)  # [B, H, W, C]

        # extract blocks from search_back using same BlockEmbeddingTF
        patches, info2 = self.block_embedding.extract(search_back)  # [B, N, ph, pw, C]
        # flatten per-block to vectors
        patches_flat = tf.reshape(patches, (B * N, -1))  # [B*N, L]

        # we expect B==1 and B*N == num_blocks, but handle general case
        # compute pairwise squared L2 distances between patches_flat (M x L) and block_query (Q x L)
        pf = tf.cast(patches_flat, tf.float32)  # [M, L]
        bq = tf.cast(self.block_query, tf.float32)  # [Q, L]
        # compute squared norms
        pf_sq = tf.reduce_sum(tf.square(pf), axis=1, keepdims=True)  # [M,1]
        bq_sq = tf.reduce_sum(tf.square(bq), axis=1, keepdims=True)  # [Q,1]
        inner = tf.matmul(pf, bq, transpose_b=True)  # [M, Q]
        dists = pf_sq - 2.0 * inner + tf.transpose(bq_sq)  # [M, Q]

        nearest_idx = tf.argmin(dists, axis=1, output_type=tf.int32)  # [M]

        # For match test we assume order of extracted blocks matches block_query order.
        # If B==1 and M == Q == num_blocks: compare indices directly.
        M = tf.shape(nearest_idx)[0]
        own_idx = tf.range(M, dtype=tf.int32)
        match_mask = tf.equal(nearest_idx, own_idx)  # [M] boolean
        match_vec_all = tf.cast(match_mask, tf.float32)  # [M]

        # If B>1: we may want to reduce to per-original-block (take first B block per original block position)
        # But for GT-HAD typical use B==1; so produce shape [num_blocks]
        # If M > self.num_blocks, take first self.num_blocks entries
        match_vec = match_vec_all[:self.num_blocks]
        # If shorter pad with zeros
        mlen = tf.shape(match_vec)[0]
        if mlen < self.num_blocks:
            pad_len = self.num_blocks - mlen
            match_vec = tf.concat([match_vec, tf.zeros([pad_len], dtype=tf.float32)], axis=0)

        return match_vec  # float tensor shape [num_blocks]

# helper to accumulate: use numpy for easy assignment (cheap) and convert to tf when needed
import numpy as np
def create_search_matrix(num_blocks, ph, pw, C, dtype=np.float32):
    return np.zeros((num_blocks, ph, pw, C), dtype=dtype)

# during training, when you get a batch `net_out` of shape [B, ph, pw, C] for indices batch_idx (tf.Tensor)
# convert net_out to numpy if using numpy buffer:
def accumulate_search_matrix_numpy(search_matrix, batch_idx, net_out):
    """
    search_matrix: numpy array [num_blocks, ph, pw, C]
    batch_idx: 1-D numpy array of indices (len B)
    net_out: tf.Tensor or numpy array [B, ph, pw, C]
    """
    if isinstance(net_out, tf.Tensor):
        arr = net_out.numpy()
    else:
        arr = net_out
    for k, idx in enumerate(batch_idx):
        search_matrix[int(idx)] = arr[k]
    return search_matrix

# before search_flag fold:
# search_matrix_batched = np.expand_dims(search_matrix, axis=0)  # [1, N, ph, pw, C]
# convert to tf before passing to BlockSearchTF
