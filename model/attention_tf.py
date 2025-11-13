import tensorflow as tf

class AttentionBlock(tf.keras.Layer):
    def __init__(self, embed_dim, patch_size=3, patch_stride=3, proj_ratio=4, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.psize = patch_size
        self.pstride = patch_stride
        # derived
        self.N = self.pstride * self.pstride   # number of patch positions inside a block
        self.P = self.psize * self.psize       # tokens per patch

        if embed_dim % proj_ratio != 0:
            raise ValueError("embed_dim should be divisible by proj_ratio")
        
        self.proj_dim = embed_dim // proj_ratio
        # projection for Q/K: maps token channels C -> proj_dim
        self.qk_proj = tf.keras.layers.Dense(self.proj_dim, use_bias=True, name='qk_proj')
        # dropout for attn (optional)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        # base mask with diagonal = large negative
        neg_inf = -1e9
        # we'll create base mask variable in build (we need accurate dtype)
        self.neg_inf = tf.constant(neg_inf, dtype=tf.float32)

    def build(self, input_shape):
        # input_shape: [B, H, W, C]
        if len(input_shape) != 4:
            raise ValueError("Input must have shape [B,H,W,C]")
        
        self.C = int(input_shape[-1])
        # base mask shape [N, N]
        base = tf.eye(self.N, dtype=tf.float32)
        base = tf.where(base > 0, tf.fill([self.N, self.N], self.neg_inf), tf.zeros([self.N, self.N], tf.float32))
        # store as constant
        self.base_mask = tf.Variable(initial_value=base, trainable=False, name='base_mask')
        super().build(input_shape)

    def call(self, x, block_idx=None, match_vec=None, return_attn=False, training=False):
        """
        x: [B, H, W, C]   (H and W must equal self.psize * self.pstride)
        block_idx: [B] Tensor of global block indices to index into match_vec. (int)
        match_vec: 1-D Tensor of length num_blocks containing 0/1 flags (float or int)
        return_attn: if True, returns attn_weights for debugging (shape [B, N, N])
        """
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self.C

        # Expect H == psize*pstride and W == psize*pstride for this internal block splitting
        expected = self.psize * self.pstride
        tf.debugging.assert_equal(H, expected, message="H must equal psize * pstride (block spatial size)")
        tf.debugging.assert_equal(W, expected, message="W must equal psize * pstride (block spatial size)")

        # Reshape: [B, pstride, psize, pstride, psize, C]
        x_view = tf.reshape(x, (B, self.pstride, self.psize, self.pstride, self.psize, C))
        # reorder to grid-first: [B, pstride, pstride, psize, psize, C]
        x_view = tf.transpose(x_view, perm=[0, 1, 3, 2, 4, 5])
        # now collapse patch-grid positions into N: [B, N, P, C]
        x_grid = tf.reshape(x_view, (B, self.N, self.P, C))  # N = pstride*pstride, P = psize*psize

        # Q/K projection: apply Dense to last dim (C->proj_dim) token-wise
        # proj_tokens: [B, N, P, proj_dim]
        proj_tokens = self.qk_proj(x_grid)  # Dense acts on last dim
        # flatten per-patch to get patch-representation vector: [B, N, P * proj_dim]
        q_repr = tf.reshape(proj_tokens, (B, self.N, self.P * self.proj_dim))
        # K same as Q (symmetric)
        k_repr = q_repr

        # Compute attention logits: [B, N, N]
        # attn_logits = q_repr @ k_repr^T
        attn_logits = tf.matmul(q_repr, k_repr, transpose_b=True)  # [B, N, N]

        # scaling
        denom = tf.cast(self.P * self.proj_dim, tf.float32)
        attn_logits = attn_logits * tf.math.rsqrt(denom)

        attn_logits = self.attn_drop(attn_logits, training=training)

        # prepare V: flatten tokens per patch (no projection) -> [B, N, P * C]
        v = tf.reshape(x_grid, (B, self.N, self.P * C))

        # Build mask batch: shape [B, N, N]
        # default: copy of base_mask for each batch
        mask_batch = tf.tile(tf.expand_dims(self.base_mask, axis=0), [B, 1, 1])  # [B, N, N]

        # If match_vec and block_idx provided, modify mask per-sample:
        if match_vec is not None and block_idx is not None:
            # gather cur_match for each batch sample
            # block_idx: int tensor of shape [B] pointing to positions in match_vec
            cur_match = tf.gather(match_vec, block_idx)  # shape [B]
            # cur_match may be float/int 0/1 -> convert to bool
            cur_match_bool = tf.cast(cur_match, tf.bool)
            # if any sample has cur_match True, for those samples we set mask to zeros (BFB)
            # Build an indices vector of samples where cur_match_bool == True
            # We'll set mask_batch[b,:,:] = 0 for each such b
            true_indices = tf.where(cur_match_bool)
            true_indices = tf.reshape(true_indices, [-1])  # shape [k]
            if tf.size(true_indices) > 0:
                # create zeros matrix for these indices
                zeros_mask = tf.zeros([tf.shape(true_indices)[0], self.N, self.N], dtype=mask_batch.dtype)
                # scatter overwrite
                mask_batch = tf.tensor_scatter_nd_update(mask_batch,
                                                         indices=tf.expand_dims(true_indices, 1),
                                                         updates=zeros_mask)
        else:
            # if not provided, keep mask_batch as base mask replicated (AFB by default)
            pass

        # Add mask to logits
        attn_logits_masked = attn_logits + mask_batch  # broadcasting ok

        # Softmax to get attn weights [B, N, N]
        attn_weights = tf.nn.softmax(attn_logits_masked, axis=-1)

        # Weighted sum: [B, N, N] @ [B, N, P*C] -> we need batch matmul per sample
        # Use einsum: out = einsum('b i j, b j k -> b i k', attn_weights, v)
        out = tf.einsum('bij,bjk->bik', attn_weights, v)  # [B, N, P*C]

        # reshape back to grid form -> [B, pstride, pstride, psize, psize, C]
        out_grid = tf.reshape(out, (B, self.pstride, self.pstride, self.psize, self.psize, C))
        # transpose back to original ordering: inverse of earlier transpose
        out_grid = tf.transpose(out_grid, perm=[0, 1, 3, 2, 4, 5])  # [B, pstride, psize, pstride, psize, C]
        # final reshape to [B, H, W, C] where H = pstride*psize
        H_out = self.pstride * self.psize
        W_out = self.pstride * self.psize
        out_final = tf.reshape(out_grid, (B, H_out, W_out, C))

        if return_attn:
            return out_final, attn_weights
        
        return out_final
