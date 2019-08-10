import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The architecture of generator is a modified U-Net.
Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
There are skip connections between the encoder and decoder (as in U-Net).
"""


class EncoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, apply_batchnorm=True):
        super(EncoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        # 进行下采样操作, strides=2
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2,
                             padding='same', kernel_initializer=initializer, use_bias=False)
        ac = layers.LeakyReLU()
        self.encoder_layer = None
        if apply_batchnorm:
            bn = layers.BatchNormalization()
            self.encoder_layer = tf.keras.Sequential([conv, bn, ac])
        else:
            self.encoder_layer = tf.keras.Sequential([conv, ac])

    def call(self, x):
        return self.encoder_layer(x)


class DecoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, apply_dropout=False):
        super(DecoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        # 上采样操作
        dconv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2,
                                       padding='same', kernel_initializer=initializer, use_bias=False)
        bn = layers.BatchNormalization()
        ac = layers.ReLU()
        self.decoder_layer = None
        if apply_dropout:
            drop = layers.Dropout(rate=0.5)
            self.decoder_layer = tf.keras.Sequential([dconv, bn, drop, ac])
        else:
            self.decoder_layer = tf.keras.Sequential([dconv, bn, ac])

    def call(self, x):
        return self.decoder_layer(x)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        encoder_layer_1 = EncoderLayer(filters=64,  kernel_size=4, apply_batchnorm=False)   # (bs, 128, 128, 64)
        encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)       # (bs, 64, 64, 128)
        encoder_layer_3 = EncoderLayer(filters=256, kernel_size=4)       # (bs, 32, 32, 256)
        encoder_layer_4 = EncoderLayer(filters=512, kernel_size=4)       # (bs, 16, 16, 512)
        encoder_layer_5 = EncoderLayer(filters=512, kernel_size=4)       # (bs, 8, 8, 512)
        encoder_layer_6 = EncoderLayer(filters=512, kernel_size=4)       # (bs, 4, 4, 512)
        encoder_layer_7 = EncoderLayer(filters=512, kernel_size=4)       # (bs, 2, 2, 512)
        encoder_layer_8 = EncoderLayer(filters=512, kernel_size=4)       # (bs, 1, 1, 512)
        self.encoder_layers = [encoder_layer_1, encoder_layer_2, encoder_layer_3, encoder_layer_4,
                               encoder_layer_5, encoder_layer_6, encoder_layer_7, encoder_layer_8]

        # U-Net. 要进行残差连接, 最后一维是指残差连接后的维度
        decoder_layer_1 = DecoderLayer(filters=512, kernel_size=4, apply_dropout=True)   # (bs, 2, 2, 1024)
        decoder_layer_2 = DecoderLayer(filters=512, kernel_size=4, apply_dropout=True)   # (bs, 4, 4, 1024)
        decoder_layer_3 = DecoderLayer(filters=512, kernel_size=4, apply_dropout=True)   # (bs, 8, 8, 1024)
        decoder_layer_4 = DecoderLayer(filters=512, kernel_size=4)   # (bs, 16, 16, 1024)
        decoder_layer_5 = DecoderLayer(filters=256, kernel_size=4)   # (bs, 32, 32, 512)
        decoder_layer_6 = DecoderLayer(filters=128, kernel_size=4)   # (bs, 64, 64, 256)
        decoder_layer_7 = DecoderLayer(filters=64, kernel_size=4)    # (bs, 128, 128, 128)
        self.decoder_layers = [decoder_layer_1, decoder_layer_2, decoder_layer_3, decoder_layer_4,
                               decoder_layer_5, decoder_layer_6, decoder_layer_7]

        # last. output.shape = (bs, 256, 256, 3)
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        self.last = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')

    def call(self, x):
        # pass the encoder and record xs
        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_xs.append(x)
        encoder_xs = encoder_xs[:-1][::-1]    # reverse
        assert len(encoder_xs) == 7

        # pass the decoder and apply skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            x = tf.concat([x, encoder_xs[i]], axis=-1)     # skip connect

        return self.last(x)        # last


# if __name__ == "__main__":
#     PATH = "data/facades/train/100.jpg"
#     image = tf.io.read_file(PATH)
#     image = tf.image.decode_jpeg(image)
#     w = tf.shape(image)[1]
#     w = w // 2
#     inp = tf.cast(image[:, :w, :], tf.float32)     # 真实图, shape=(256,256,3)
#     re = tf.cast(image[:, w:, :], tf.float32)    # 草图, shape=(256,256,3)
#
#     # model
#     generator = Generator()
#     gen_output = generator(inp[tf.newaxis, ...])
#     print(gen_output.shape)
#     print(gen_output.numpy().max(), gen_output.numpy().min())
#     plt.imshow(gen_output[0, ...].numpy().astype("float")+1.0/2.0)
#     plt.colorbar()
#     plt.show()
#     generator.summary()
