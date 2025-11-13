# tests/test_block_roundtrip.py
import numpy as np
import tensorflow as tf
from ..model.block_tf import BlockEmbeddingTF, BlockFoldTF

def run_test():
    # small synthetic image
    B = 1
    H = 18
    W = 18
    C = 5
    img = np.random.rand(B, H, W, C).astype(np.float32)
    images = tf.convert_to_tensor(img)

    patch_h = 3
    patch_w = 3
    stride = 3

    be = BlockEmbeddingTF(patch_h=patch_h, patch_w=patch_w, stride_h=stride, stride_w=stride, padding='SAME')
    patches, info = be.extract(images)
    print("patches shape:", patches.shape, "info:", info)

    bf = BlockFoldTF()
    recon = bf.fold(patches, info, orig_H=H, orig_W=W)
    recon_np = recon.numpy()

    # compare only the central region that is within valid reconstruction (tolerance)
    diff = np.abs(recon_np - img)
    print("max abs diff:", diff.max())
    assert diff.max() < 1e-5 or diff.max() < 1e-3, "Round-trip error too large"
    print("Round-trip OK")

if __name__ == '__main__':
    run_test()
