import keras
from keras import layers

from models.customs import EqualizedConv, EqualizedDense, fade_in


class Discriminator:
    def __init__(self, begin_scale_log2, max_scale_log2):
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

        self.rgb_in_model = []
        self.d_blocks_model = []

        for i in range(begin_scale_log2, max_scale_log2 + 1):
            self.rgb_in_model.append(self.build_rgb_in(i, self.filter_nums[i]))
            if i == begin_scale_log2:
                begin_scale = 2 ** begin_scale_log2
                self.d_blocks_model.append(self.build_base(begin_scale, self.filter_nums[begin_scale_log2]))
            else:
                self.d_blocks_model.append(self.build_block(i, self.filter_nums[i], self.filter_nums[i-1]))


    def build_base(self, begin_scale, filter_num):
        input_tensor = layers.Input(shape=(begin_scale, begin_scale, filter_num))
        x = layers.BatchNormalization()(input_tensor)
        x = EqualizedConv(filter_num, 3, name="d_equalized_conv_base")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Flatten()(x)
        x = EqualizedDense(200, name="d_equalized_dense_200")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = EqualizedDense(1, name="d_equalized_dense_1")(x)
        return keras.Model(input_tensor, x, name="d_base")


    def build_rgb_in(self, scale_log2, filter_num):
        scale = 2**scale_log2
        rgb_in = layers.Input(shape=(scale, scale, 3), name=f"d_input_{scale}")
        x = EqualizedConv(filter_num, 1, name=f"d_equalized_conv_rgb_in_{scale_log2}")(rgb_in)
        x = layers.LeakyReLU(0.2)(x)
        return keras.Model(rgb_in, x)


    def build_block(self, scale_log2, filter_num, filter_num2):
        scale = 2 ** scale_log2
        input_tensor = layers.Input(shape=(scale, scale, filter_num))
        x = EqualizedConv(filter_num, 3, name=f"d_equalized_conv_{scale_log2}_0")(input_tensor)
        x = layers.LeakyReLU(0.2)(x)
        x = EqualizedConv(filter_num2, 3, name=f"d_equalized_conv_{scale_log2}_1")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D()(x)
        return keras.Model(input_tensor, x)


    def get_model(self, target_scale_log2):
        scale = 2 ** target_scale_log2
        input_tensor = layers.Input(shape=(scale, scale, 3))
        alpha = layers.Input(shape=(1, ), name="g_alpha")
        x = self.rgb_in_model[target_scale_log2 - 2](input_tensor)
        x = self.d_blocks_model[target_scale_log2 - 2](x)

        if target_scale_log2 > self.begin_scale_log2:
            down_sample = layers.AveragePooling2D()(input_tensor)
            y = self.rgb_in_model[target_scale_log2 - 3](down_sample)
            x = fade_in(alpha[0], x, y)

        for i in range(target_scale_log2 - 3, -1, -1):
            x = self.d_blocks_model[i](x)
        return keras.Model([input_tensor, alpha], x, name=f"discriminator_scale_{scale}")


if __name__ == "__main__":
    D = Discriminator(2, 8)
    d = D.get_model(2)
    d.summary()