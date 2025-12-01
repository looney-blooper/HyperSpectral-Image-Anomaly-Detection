import os
import random
import numpy as np
import tensorflow as tf
import scipy.io as sio
from sklearn.metrics import roc_auc_score, roc_curve

# === CONFIGURATION (MUST MATCH TRAIN.PY) ===
DATA_DIR = "./data"
RESULTS_DIR = "./results"
FILES = ["los-angeles-1"]  # The file you want to test
SEED = 42

# Model Hyperparams (These MUST match what you trained with)
BATCH_SIZE = 32
PATCH_SIZE = 3            
PATCH_STRIDE = 3          
BLOCK_SIZE = PATCH_SIZE * PATCH_STRIDE 
EMBED_DIM = 64
SPATIAL_SMOOTH_K = 3      

# === IMPORTS ===
# Ensure the 'model' folder is in the same directory as this script
from model.block_tf import BlockEmbeddingTF, BlockFoldTF
from model.gtblock_tf import NetTF

# -------------------------
# Utilities
# -------------------------
def normalize_image(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    return arr

def save_results(save_dir: str, residual_map: np.ndarray, roc_pd: np.ndarray, roc_pf: np.ndarray) -> None:
    os.makedirs(save_dir, exist_ok=True)
    sio.savemat(os.path.join(save_dir, "GT-HAD_inference_map.mat"), {"show": residual_map})
    print(f"Saved .mat file to {save_dir}")

# -------------------------
# Inference Routine
# -------------------------
def run_inference(filename: str):
    print(f"\n=== INFERENCE: {filename} ===")
    
    # 1. Load and Preprocess Data
    mat_path = os.path.join(DATA_DIR, filename + ".mat")
    if not os.path.exists(mat_path):
        print(f"Error: File not found {mat_path}")
        return

    mat = sio.loadmat(mat_path)
    img_np: np.ndarray = mat["data"]          # H x W x Bands
    gt_map: np.ndarray = mat["map"]           # H x W
    H, W, bands = img_np.shape

    img_norm = normalize_image(img_np)
    img_tf = tf.expand_dims(tf.convert_to_tensor(img_norm, dtype=tf.float32), axis=0)

    # 2. Extract Blocks (Same as training)
    block_embed = BlockEmbeddingTF(
        patch_h=BLOCK_SIZE, patch_w=BLOCK_SIZE,
        stride_h=PATCH_STRIDE, stride_w=PATCH_STRIDE,
        padding="SAME"
    )
    patches_batched, info = block_embed.extract(img_tf)
    patches_np = patches_batched.numpy()[0] # [N, ph, pw, C]
    num_blocks = patches_np.shape[0]
    
    # 3. Initialize Model Architecture
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

    # 4. Load Weights
    # We create a dummy Checkpoint object to match the structure we saved
    ckpt = tf.train.Checkpoint(net=net)
    
    # Locate the checkpoint folder
    ckpt_dir = os.path.join(RESULTS_DIR, filename, 'checkpoints')
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    
    if latest_ckpt:
        print(f"Loading weights from: {latest_ckpt}")
        # .expect_partial() prevents warnings about missing optimizer variables
        ckpt.restore(latest_ckpt).expect_partial()
    else:
        print(f"WARNING: No checkpoint found in {ckpt_dir}!")
        print("Running with random weights (Results will be garbage).")

    # 5. Prepare Inference Dataset
    indices = np.arange(num_blocks, dtype=np.int32)
    ds_infer = tf.data.Dataset.from_tensor_slices((patches_np, indices)).batch(BATCH_SIZE)

    # 6. Run Prediction Loop
    print("Running forward pass...")
    residual_blocks = []
    
    # Initialize a dummy match_vec. 
    # In pure inference, we assume the gating unit handles the decision, 
    # or we start with a neutral vector.
    match_vec_dummy = tf.zeros([num_blocks], dtype=tf.float32)

    for (in_batch_np, idx_batch_np) in ds_infer:
        x_in = tf.convert_to_tensor(in_batch_np, dtype=tf.float32)
        idx_batch = tf.convert_to_tensor(idx_batch_np, dtype=tf.int32)
        
        # Forward pass with training=False
        out = net(x_in, block_idx=idx_batch, match_vec=match_vec_dummy, training=False)

        # Calculate Residuals (Squared Error)
        res = tf.square(x_in - out)               # [B, ph, pw, C]
        
        # Average across spectral bands
        res_spec_avg = tf.reduce_mean(res, axis=-1, keepdims=True) # [B, ph, pw, 1]

        # Spatial Smoothing (Avg Pool)
        res_smoothed = tf.nn.avg_pool2d(res_spec_avg, ksize=SPATIAL_SMOOTH_K, strides=1, padding="SAME")
        residual_blocks.append(res_smoothed)

    # 7. Reconstruct the Anomaly Map
    residual_blocks_all = tf.concat(residual_blocks, axis=0)
    search_matrix_res = tf.expand_dims(residual_blocks_all, axis=0)
    
    block_fold = BlockFoldTF()
    recon_residual = block_fold.fold(search_matrix_res, info, orig_H=H, orig_W=W)
    residual_map = recon_residual.numpy()[0, :, :, 0]

    # Normalize Map
    residual_map -= residual_map.min()
    if residual_map.max() > 0:
        residual_map /= residual_map.max()

    # 8. Calculate Metrics (AUC)
    auc = roc_auc_score(gt_map.flatten().astype(int), residual_map.flatten())
    print(f"inference AUC = {auc:.4f}")

    # 9. Save Results
    out_dir = os.path.join(RESULTS_DIR, filename, 'inference_output')
    save_results(out_dir, residual_map, None, None)

if __name__ == "__main__":
    for f in FILES:
        run_inference(f)