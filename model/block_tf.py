import tensorflow as tf

class BlockEmbeddingTF:
    def __init__(self, patch_h=9, patch_w=9, stride_h=3, stride_w=3, padding='SAME'):
        self.ph = patch_h
        self.pw = patch_w
        self.sh = stride_h
        self.sw = stride_w
        self.padding = padding  # 'SAME' or 'VALID'

    def extract(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1,self.ph,self.pw,1],
            strides=[1,self.sh,self.sw,1],
            rates=[1,1,1,1],
            padding=self.padding,
        )

        Batch_size = tf.shape(patches)[0]
        new_h = tf.shape(patches)[1]
        new_w = tf.shape(patches)[2]
        flat_dim = int(self.ph * self.pw)
        C = tf.shape(images)[-1]

        patches = tf.reshape(patches, (Batch_size, new_h, new_w, self.ph, self.pw, -1))

        C_orig = images.shape[-1]

        patches = tf.reshape(patches, (Batch_size, new_h, new_w, self.ph, self.pw, C_orig)) 
        N = new_h * new_w
        patches = tf.reshape(patches, (Batch_size, N, self.ph, self.pw, C_orig))
        info = {'new_h': int(new_h.numpy()), 'new_w': int(new_w.numpy()),
                'ph': self.ph, 'pw': self.pw, 'sh': self.sh, 'sw': self.sw,
                'padding': self.padding}
        return patches, info

class BlockFoldTF:
    def __init__(self):
        pass

    def fold(self, patches, info, orig_H=None, orig_W=None):
        """
        Overlap-aware vectorized fold using tf.scatter_nd.
        patches: [B, N, ph, pw, C]
        info: dict with 'new_h','new_w','ph','pw','sh','sw','padding'
        returns: [B, orig_H, orig_W, C]  (if orig_H/orig_W provided) else [B, new_h*ph, new_w*pw, C]
        """
        patches = tf.convert_to_tensor(patches)
        B = tf.shape(patches)[0]
        ph = int(info['ph']); pw = int(info['pw'])
        new_h = int(info['new_h']); new_w = int(info['new_w'])
        sh = int(info['sh']); sw = int(info['sw'])
        C = tf.shape(patches)[-1]

        # reshape patches to grid [B, new_h, new_w, ph, pw, C]
        patches_grid = tf.reshape(patches, (B, new_h, new_w, ph, pw, C))

        # create index grids
        i = tf.range(new_h, dtype=tf.int32)
        j = tf.range(new_w, dtype=tf.int32)
        u = tf.range(ph, dtype=tf.int32)
        v = tf.range(pw, dtype=tf.int32)
        I, J, U, V = tf.meshgrid(i, j, u, v, indexing='ij')  # shapes [new_h,new_w,ph,pw]

        # destination coords
        Y = I * sh + U   # [new_h,new_w,ph,pw]
        X = J * sw + V   # [new_h,new_w,ph,pw]

        # Clip to orig dims if provided, otherwise compute target sizes
        if orig_H is None:
            H_out = new_h * ph
        else:
            H_out = orig_H
        if orig_W is None:
            W_out = new_w * pw
        else:
            W_out = orig_W

        Y = tf.clip_by_value(Y, 0, H_out - 1)
        X = tf.clip_by_value(X, 0, W_out - 1)

        # flatten coords (Ucount = new_h*new_w*ph*pw)
        Y_flat = tf.reshape(Y, (-1,))  # [Ucount]
        X_flat = tf.reshape(X, (-1,))  # [Ucount]

        # flatten patch values -> [B, Ucount, C]
        patches_flat = tf.reshape(patches_grid, (B, -1, C))  # [B, Ucount, C]

        # prepare scatter indices for all batches and channels
        B_range = tf.range(B, dtype=tf.int32)
        b_idx = tf.repeat(B_range, repeats=tf.shape(patches_flat)[1])  # [B*Ucount]
        y_idx = tf.tile(Y_flat, [B])  # [B*Ucount]
        x_idx = tf.tile(X_flat, [B])  # [B*Ucount]

        patches_flat2d = tf.reshape(patches_flat, (-1, C))  # [B*Ucount, C]

        # Expand per-channel
        b_rep_per_channel = tf.repeat(b_idx, repeats=C)   # [B*Ucount*C]
        y_rep_per_channel = tf.repeat(y_idx, repeats=C)
        x_rep_per_channel = tf.repeat(x_idx, repeats=C)
        ch_idx = tf.tile(tf.range(C, dtype=tf.int32), [tf.shape(patches_flat2d)[0]])  # [B*Ucount*C]

        values = tf.reshape(patches_flat2d, (-1,))  # [B*Ucount*C]

        indices = tf.stack([b_rep_per_channel, y_rep_per_channel, x_rep_per_channel, ch_idx], axis=1)  # [M,4]
        out_shape = (B, H_out, W_out, C)

        recon = tf.scatter_nd(indices, values, out_shape)
        counts = tf.scatter_nd(indices, tf.ones_like(values, dtype=tf.float32), out_shape)
        counts_nonzero = tf.where(counts == 0.0, tf.ones_like(counts), counts)
        recon_avg = recon / counts_nonzero

        return recon_avg