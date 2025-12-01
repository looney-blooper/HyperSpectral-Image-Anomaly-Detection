import tensorflow as tf
from model.gtblock_tf import NetTF

# Define the shapes (Must match your config)
bands = 205         # Example for standard HSI, change to your data's band count
BLOCK_SIZE = 9      # Patch Size (3) * Stride (3)
EMBED_DIM = 64
NUM_BLOCKS = 100    # Dummy number

# 1. Initialize Model
net = NetTF(
    in_chans=bands,
    embed_dim=EMBED_DIM,
    patch_size=3,
    patch_stride=3,
    mlp_ratio=2.0,
    attn_drop=0.0,
    drop=0.0,
    proj_ratio=4
)

# 2. Create Dummy Inputs
# The model expects: Input Image Patch, Block Indices, and Match Vector
dummy_input = tf.zeros((1, BLOCK_SIZE, BLOCK_SIZE, bands), dtype=tf.float32)
dummy_idx = tf.zeros((1,), dtype=tf.int32)
dummy_match = tf.zeros((NUM_BLOCKS,), dtype=tf.float32)

# 3. Build (Forward Pass)
# We must run it once so TF figures out the shapes
_ = net(dummy_input, block_idx=dummy_idx, match_vec=dummy_match)

# 4. Print Summary
print("\n=== MODEL SUMMARY ===")
net.summary()

# Optional: Print total params specifically
total_params = net.count_params()
print(f"\nTotal Trainable Parameters: {total_params:,}")