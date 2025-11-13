import tensorflow as tf
from model.attention_tf import AttentionBlock


class MLP(tf.keras.Layer):
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(dim * mlp_ratio)
        self.fc1 = tf.keras.layers.Dense(self.hidden_dim, name="mlp_fc1")
        self.act = tf.keras.layers.Activation('gelu')
        self.fc2 = tf.keras.layers.Dense(dim, name="mlp_fc2")
        self.dropout = tf.keras.layers.Dropout(drop)

    def call(self, x, training = False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x
    
class TransformerBlock(tf.keras.Layer):
    def __init__(self, embed_dim, patch_size=3, patch_stride=3, mlp_ratio=2.0, attn_drop=0.0, drop=0.0, proj_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="lnorm1")
        self.attn = AttentionBlock(embed_dim, patch_size, patch_stride, proj_ratio, attn_drop, name="attention")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="lnorm2")
        self.mlp = MLP(embed_dim, mlp_ratio, drop, name="mlp")

    def call(self, x, block_idx=None, match_vec=None,  training=False):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        x_flat = tf.reshape(x, (B, H*W, C))
        x_norm = self.norm1(x_flat)
        x_norm = tf.reshape(x_norm, (B,H,W,C))

        attn_out = self.attn(x_norm, block_idx=block_idx, match_vec=match_vec, training=training)
        x1 = attn_out

        x1_flat = tf.reshape(x1, (B,H*W,C))
        x2 = self.norm2(x1_flat)

        x2 = self.mlp(x2, training=training)
        x_out = x1_flat + x2
        x_out = tf.reshape(x_out, (B,H,W,C))

        return x_out
    
class NetTF(tf.keras.Model):
    def __init__(self, in_chans, embed_dim=64, patch_size=3, patch_stride=3, mlp_ratio=2.0, attn_drop=0.0, drop=0.0, proj_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.conv_head = tf.keras.layers.Conv2D(embed_dim, kernel_size=3, padding='SAME', name='conv_head')
        self.gtb = TransformerBlock(embed_dim=embed_dim, patch_size=patch_size, patch_stride=patch_stride, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop, proj_ratio=proj_ratio)
        self.conv_tail = tf.keras.layers.Conv2D(in_chans, kernel_size=3, padding='SAME', name='conv_tail')


    def call(self, x, block_idx=None, match_vec=None, training=False):
        x = self.conv_head(x)
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        x = self.gtb(x, block_idx=block_idx, match_vec=match_vec, training=training)
        x = self.conv_tail(x)   
        return x

