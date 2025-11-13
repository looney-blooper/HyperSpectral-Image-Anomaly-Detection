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
        patches: [B, N, ph, pw, C]
        info: dict with keys 'new_h','new_w','ph','pw','sh','sw','padding'
        returns: [B, new_h*ph, new_w*pw, C]
        NOTE: This implementation assumes no overlap: sh == ph and sw == pw.
        """
        ph = info['ph']; pw = info['pw']
        new_h = info['new_h']; new_w = info['new_w']

        # dynamic dims
        B = tf.shape(patches)[0]
        C = tf.shape(patches)[-1]

        # reshape to [B, new_h, new_w, ph, pw, C]
        x = tf.reshape(patches, (B, new_h, new_w, ph, pw, C))

        # transpose to [B, new_h, ph, new_w, pw, C]
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])

        # final reshape to [B, new_h*ph, new_w*pw, C]
        out = tf.reshape(x, (B, new_h * ph, new_w * pw, C))

        # optionally crop to orig_H/orig_W if provided and padding was used
        if orig_H is not None and orig_W is not None:
            out = out[:, :orig_H, :orig_W, :]
        return out
