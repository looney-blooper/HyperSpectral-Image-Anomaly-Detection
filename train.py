import os
import time
import random
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
import scipy.io as sio
from sklearn.metrics import roc_auc_score, roc_curve

# === adjust these paths/flags as needed ===
DATA_DIR = "./data"
RESULTS_DIR = "./results"
FILES = ["los-angeles-1"]        # list of mat filenames (without .mat)
SEED = 42

# training hyperparams
BATCH_SIZE = 32
END_ITER = 150
SEARCH_ITER = 25
LR = 1e-3

# model / block params (must match your implementations)
PATCH_SIZE = 3            # psize (mini patch side inside block)
PATCH_STRIDE = 3          # pstride (mini-patch grid count per axis)
BLOCK_SIZE = PATCH_SIZE * PATCH_STRIDE   # sliding window (block) size
EMBED_DIM = 64

# residual smoothing (approximation)
SPATIAL_SMOOTH_K = 3  # 2D avg-pool kernel size after spectral averaging

# === imports from your code (adjust module paths if you moved files) ===
from model.block_tf import BlockEmbeddingTF, BlockFoldTF             # channels-last extractor/fold
from model.block_search_tf import BlockSearchTF                      # CMM
from model.gtblock_tf import NetTF                                   # conv_head -> GTB -> conv_tail


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize HxWxC image to [0,1] float32."""
    arr = img.astype(np.float32)
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    return arr


def save_results(save_dir: str, residual_map: np.ndarray, roc_pd: np.ndarray, roc_pf: np.ndarray) -> None:
    os.makedirs(save_dir, exist_ok=True)
    sio.savemat(os.path.join(save_dir, "GT-HAD_map.mat"), {"show": residual_map})
    sio.savemat(os.path.join(save_dir, "GT-HAD_roc.mat"), {"PD": roc_pd, "PF": roc_pf})


# -------------------------
# Core training routine
# -------------------------
def train_on_file(filename: str) -> None:
    """
    Train GT-HAD on a single file (MAT containing 'data' [H,W,Bands] and 'map' ground truth).
    Produces saved result files and prints AUC.
    """
    print(f"\n=== TRAIN: {filename} ===")
    set_seed(SEED)

    # ----- load data -----
    mat = sio.loadmat(os.path.join(DATA_DIR, filename + ".mat"))
    img_np: np.ndarray = mat["data"]          # H x W x Bands
    gt_map: np.ndarray = mat["map"]           # H x W

    H, W, bands = img_np.shape
    print(f"Loaded {filename}: shape={img_np.shape}")

    img_norm = normalize_image(img_np)
    img_tf = tf.expand_dims(tf.convert_to_tensor(img_norm, dtype=tf.float32), axis=0)  # [1,H,W,C]

    # ----- block extractor & dataset -----
    block_embed = BlockEmbeddingTF(
        patch_h=BLOCK_SIZE, patch_w=BLOCK_SIZE,
        stride_h=PATCH_STRIDE, stride_w=PATCH_STRIDE,
        padding="SAME"
    )

    patches_batched, info = block_embed.extract(img_tf)   # [1, N, ph, pw, C]
    patches_np: np.ndarray = patches_batched.numpy()[0]    # [N, ph, pw, C]
    num_blocks = patches_np.shape[0]
    ph, pw, C = patches_np.shape[1], patches_np.shape[2], patches_np.shape[3]

    print(f"Num blocks: {num_blocks}, patch shape: {ph}x{pw}x{C}")

    # block_query for BlockSearch (flattened)
    block_query = tf.reshape(patches_batched[0], (num_blocks, -1))   # [N, L]

    # build dataset: (gt_block, input_block, index)
    indices = np.arange(num_blocks, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((patches_np, patches_np, indices))
    ds = ds.shuffle(buffer_size=num_blocks, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ----- model / optimizer / loss -----
    net = NetTF(
        in_chans=bands,
        embed_dim=EMBED_DIM,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        mlp_ratio=2.0,
        attn_drop=0.0,
        drop=0.0,
        proj_ratio=4
    )

    # build model once (call with dummy)
    dummy_in = tf.zeros((1, BLOCK_SIZE, BLOCK_SIZE, bands), dtype=tf.float32)
    net(dummy_in, block_idx=tf.constant([0], dtype=tf.int32), match_vec=tf.zeros([num_blocks], dtype=tf.float32))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    mse_loss = tf.keras.losses.MeanSquaredError()

    # ----- state buffers -----
    # search_matrix: numpy buffer used to accumulate reconstructions when search_flag is set
    search_matrix = np.zeros((num_blocks, ph, pw, C), dtype=np.float32)
    match_vec = tf.zeros([num_blocks], dtype=tf.float32)   # 0/1 flags (AFB/BFB) updated periodically

    block_fold = BlockFoldTF()
    block_search = BlockSearchTF(block_embedding=block_embed, block_query=block_query)

    # ----- training loop -----
    start_time = time.time()
    for itr in range(1, END_ITER + 1):
        search_flag = (itr % SEARCH_ITER == 0 and itr != END_ITER)
        epoch_loss = 0.0
        step_count = 0

        for (gt_batch_np, in_batch_np, idx_batch_np) in ds:
            # convert to tensors
            x_in = tf.convert_to_tensor(in_batch_np, dtype=tf.float32)   # [B,ph,pw,C]
            idx_batch = tf.convert_to_tensor(idx_batch_np, dtype=tf.int32)

            # single train step (eager; wrap in tf.function later if desired)
            with tf.GradientTape() as tape:
                out = net(x_in, block_idx=idx_batch, match_vec=match_vec, training=True)  # [B,ph,pw,C]
                loss = mse_loss(tf.convert_to_tensor(gt_batch_np, dtype=tf.float32), out)
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

            epoch_loss += float(loss.numpy())
            step_count += 1

            # accumulate outputs into search_matrix when search_flag is True
            if search_flag:
                out_np = out.numpy()
                for local_i, global_idx in enumerate(idx_batch_np):
                    search_matrix[int(global_idx)] = out_np[local_i]

        avg_loss = epoch_loss / max(1, step_count)
        print(f"Iter {itr:3d}/{END_ITER:3d}  loss={avg_loss:.6f}  search_flag={search_flag}")

        # perform CMM search & update match_vec when search_flag
        if search_flag:
            # convert search_matrix to batched tf tensor and fold back to image
            search_batched = tf.convert_to_tensor(np.expand_dims(search_matrix, axis=0), dtype=tf.float32)  # [1,N,ph,pw,C]
            search_back = block_fold.fold(search_batched, info, orig_H=H, orig_W=W)  # [1,H,W,C]
            # compute match_vec (1 for blocks that match themselves)
            match_vec = block_search.compute_match_vec_from_batched_search_matrix(search_batched, info, orig_H=H, orig_W=W)
            match_sum = int(tf.reduce_sum(match_vec).numpy())
            print(f"  -> Updated match_vec (sum={match_sum}/{num_blocks})")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    # ----- inference (final) -----
    print("Running final inference and computing AUC...")
    ds_infer = tf.data.Dataset.from_tensor_slices((patches_np, patches_np, indices)).batch(BATCH_SIZE)
    residual_blocks = []  # will store [B, ph, pw, 1] blocks

    for (_, in_batch_np, idx_batch_np) in ds_infer:
        x_in = tf.convert_to_tensor(in_batch_np, dtype=tf.float32)
        idx_batch = tf.convert_to_tensor(idx_batch_np, dtype=tf.int32)
        out = net(x_in, block_idx=idx_batch, match_vec=match_vec, training=False)

        # per-pixel squared residuals, average across spectral bands
        res = tf.square(x_in - out)                      # [B, ph, pw, C]
        res_spec_avg = tf.reduce_mean(res, axis=-1, keepdims=True)  # [B, ph, pw, 1]

        # simple spatial smoothing (2D avg pool)
        # tf.nn.avg_pool expects NHWC: [B, H, W, C]
        res_smoothed = tf.nn.avg_pool2d(res_spec_avg, ksize=SPATIAL_SMOOTH_K, strides=1, padding="SAME")  # [B, ph, pw, 1]
        residual_blocks.append(res_smoothed)

    # collect blocks and fold back into full residual map
    residual_blocks_all = tf.concat(residual_blocks, axis=0)  # [N, ph, pw, 1]
    search_matrix_res = tf.expand_dims(residual_blocks_all, axis=0)  # [1, N, ph, pw, 1]
    recon_residual = block_fold.fold(search_matrix_res, info, orig_H=H, orig_W=W)  # [1,H,W,1]
    residual_map = recon_residual.numpy()[0, :, :, 0]

    # normalize residual map to 0..1
    residual_map -= residual_map.min()
    if residual_map.max() > 0:
        residual_map /= residual_map.max()

    # compute AUC
    auc = roc_auc_score(gt_map.flatten().astype(int), residual_map.flatten())
    fpr, tpr, _ = roc_curve(gt_map.flatten().astype(int), residual_map.flatten())
    print(f"Final AUC = {auc:.4f}")

    # save outputs
    out_dir = os.path.join(RESULTS_DIR, filename)
    save_results(out_dir, residual_map, tpr, fpr)
    print(f"Saved results to {out_dir}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for f in FILES:
        train_on_file(f)
