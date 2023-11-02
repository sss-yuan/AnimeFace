import keras
from keras import layers
import tensorflow as tf

from models.customs import pixel_norm, EqualizedDense, EqualizedConv, AddNoise, AdaIN, fade_in


def Build_Mapping(input_shape=512):
    z = layers.Input(shape=(input_shape))
    w = pixel_norm(z)
    for i in range(6):
        w = EqualizedDense(input_shape, learning_rate_multiplier=0.01, name=f'mapping_equalized_dense_{i}')(w)
        w = layers.LeakyReLU(alpha=0.2)(w)
    # w = tf.tile(tf.expand_dims(w, 1), (1, num_stages, 1))
    return keras.Model(z, w, name="mapping")


class Generator:
    def __init__(self, begin_scale_log2, max_scale_log2, z_dim=512):
        self.begin_scale_log2 = begin_scale_log2
        self.filter_nums = {
            2: 512,  # 4x4
            3: 512,  # 8x8
            4: 256,  # 16x16
            5: 128,  # 32x32
            6: 64,  # 64x64
            7: 32,  # 128x128
            8: 16,  # 256x256
        }

        self.w = layers.Input(shape=(z_dim,))
        self.noise_input = [layers.Input(shape=(2**i, 2**i, 1), name=f'noise_{2**i}') for i in range(begin_scale_log2, max_scale_log2 + 1)]

        self.g_blocks_out = []
        self.to_rgb_out = []

        self.g_input = layers.Input(shape=(2 ** begin_scale_log2, 2 ** begin_scale_log2, self.filter_nums[begin_scale_log2]))
        g = self.g_input
        for i in range(begin_scale_log2, max_scale_log2 + 1):
            g = self.build_block_out(g, i-2, self.filter_nums[i], i == begin_scale_log2)
            self.g_blocks_out.append(g)
            self.to_rgb_out.append(self.build_rgb_out(g, i))

    def build_block_out(self, input_tensor, scale_index, filter_num, is_base):
        noise = self.noise_input[scale_index]
        x = input_tensor

        if not is_base:
            x = layers.UpSampling2D()(x)
            x = EqualizedConv(filter_num, 3, name=f"g_equalized_conv_{scale_index}_0")(x)

        x = AddNoise()([x, noise])
        x = layers.LeakyReLU(0.2)(x)
        x = AdaIN()([x, self.w])

        x = EqualizedConv(filter_num, 3, name=f"g_equalized_conv_{scale_index}_1")(x)
        x = AddNoise()([x, noise])
        x = layers.LeakyReLU(0.2)(x)
        x = AdaIN()([x, self.w])
        return x

    def build_rgb_out(self, input_tensor, i):
        x = EqualizedConv(3, 1, gain=1, name=f"g_equalized_conv_rgb_out_{i}")(input_tensor)
        return x

    def get_model(self, target_scale_log2):
        scale = 2 ** target_scale_log2

        alpha = layers.Input(shape=(1, ), name="g_alpha")
        out = self.to_rgb_out[target_scale_log2-2]
        if target_scale_log2 > self.begin_scale_log2:
            old_out = self.to_rgb_out[target_scale_log2-3]
            old_out = layers.UpSampling2D(name=f'UpSampling2D_scale_log2_{target_scale_log2-3}')(old_out)
            out = fade_in(alpha[0], out, old_out)

        model = keras.Model(
            [self.g_input, self.w, alpha, self.noise_input],
            out,
            name=f'generator_scale_{scale}'
        )
        return model


if __name__ == "__main__":
    Build_Mapping()
    G = Generator(2, 8)
    g = G.get_model(8)
    g.summary()


