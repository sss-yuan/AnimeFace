import keras
import numpy as np
from PIL import Image
from keras import layers
import tensorflow as tf
from models.Generator import Generator, Build_Mapping
from models.Discriminator import Discriminator
from models.customs import wasserstein_loss
import os


class GAN(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scales = [4, 8, 16, 32, 64, 128, 256]
        self.scales_log2 = [2, 3, 4, 5, 6, 7, 8]
        self.filter_nums = {
            2: 512,  # 4x4
            3: 512,  # 8x8
            4: 256,  # 16x16
            5: 128,  # 32x32
            6: 64,  # 64x64
            7: 32,  # 128x128
            8: 16,  # 256x256
        }
        self.z_dim = 512
        self.alpha_step = tf.Variable(0.0001, dtype=tf.float32, trainable=False, name="alpha")
        self.alpha = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="alpha")
        self.phase = None

        self.begin_scale_log2 = self.scales_log2[0]
        self.begin_scale = 2 ** self.begin_scale_log2
        self.G_base = Generator(self.scales_log2[0], self.scales_log2[-1])
        self.D_base = Discriminator(self.scales_log2[0], self.scales_log2[-1])
        self.mapping = Build_Mapping()

        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def set_scale(self, scale_log2):
        tf.keras.backend.clear_session()
        self.cur_scale_log2 = scale_log2
        self.G = self.G_base.get_model(scale_log2)
        self.D = self.D_base.get_model(scale_log2)

        scale = 2 ** scale_log2
        print(f"Model resolution:{scale}x{scale}")
        os.makedirs(self.log_dir + f"/{scale}x{scale}/G", exist_ok=True)
        os.makedirs(self.log_dir + f"/{scale}x{scale}/D", exist_ok=True)
        # os.makedirs(self.log_dir + f"/{scale}x{scale}/C", exist_ok=True)
        os.makedirs(self.log_dir + f"/{scale}x{scale}/sample", exist_ok=True)
        os.makedirs(self.log_dir + f"/{scale}x{scale}/M", exist_ok=True)
        self.alpha.assign(0.0)

    def my_save(self, post_fix):
        scale = 2 ** self.cur_scale_log2
        self.G.save_weights(self.log_dir + f"/{scale}x{scale}/G/{scale}x{scale}_{post_fix}.h5")
        self.D.save_weights(self.log_dir + f"/{scale}x{scale}/D/{scale}x{scale}_{post_fix}.h5")
        self.mapping.save_weights(self.log_dir + f"/{scale}x{scale}/M/{scale}x{scale}_{post_fix}.h5")
        # self.save_weights(self.log_dir + f"/{scale}x{scale}/C/{scale}x{scale}_{post_fix}.h5")

    def gen_sample_img(self, scale_log2, post_fix, batch_size=4):
        val_z = tf.random.normal((batch_size, self.z_dim))
        noise = [
            tf.random.normal((batch_size, 2 ** scale_log2, 2 ** scale_log2, 1))
            for scale_log2 in range(self.begin_scale_log2, self.scales_log2[-1] + 1)
        ]
        const_input = tf.ones(shape=(batch_size, self.begin_scale, self.begin_scale, self.filter_nums[self.begin_scale_log2]))
        alpha = tf.ones((batch_size, 1))

        w = self.mapping.predict(val_z)
        images = self.G.predict([const_input, w, alpha, noise])
        images = np.clip((images * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        scale = 2 ** scale_log2
        w, h, _ = images[0].shape
        grid = Image.new('RGB', size=(2 * w, 2 * h))
        for i, image in enumerate(images):
            grid.paste(Image.fromarray(image), box=(i % 2 * w, i // 2 * h))
        grid.save(self.log_dir + f"/{scale}x{scale}/sample/{scale}x{scale}_{post_fix}.png")


    def compile(
            self,
            g_optimizer,
            d_optimizer,
            scale_log2,
            *args,
            **kwargs
    ):
        self.set_scale(scale_log2)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_weights = kwargs.pop("loss_weights", {"gradient_penalty": 10, "drift": 0.001})
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

        self.d_loss_metric_fake = keras.metrics.Mean(name="d_loss_fake")
        self.d_loss_metric_real = keras.metrics.Mean(name="d_loss_real")
        self.d_loss_metric_penalty = keras.metrics.Mean(name="d_loss_penalty")
        self.d_loss_metric_drift = keras.metrics.Mean(name="d_loss_drift")

        super().compile(*args, **kwargs)

    @property
    def metrics(self):
        # return [self.d_loss_metric, self.g_loss_metric]
        return [self.d_loss_metric, self.g_loss_metric, self.d_loss_metric_fake,  self.d_loss_metric_real,  self.d_loss_metric_penalty, self.d_loss_metric_drift]

    def gradient_loss(self, gradient):
        loss = tf.square(gradient)
        loss = tf.reduce_sum(loss, axis=tf.range(1, tf.size(tf.shape(loss))))
        loss = tf.sqrt(loss)
        loss = tf.square(loss - 1)
        loss = tf.reduce_mean(loss)
        return loss

    def set_phase(self, phase):
        self.phase = phase

    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]
        real_labels = -tf.ones(batch_size)
        fake_labels = tf.ones(batch_size)

        z = tf.random.normal((batch_size, self.z_dim))
        const_input = tf.ones((batch_size,  self.begin_scale,  self.begin_scale, self.filter_nums[self.begin_scale_log2]))
        noise = [tf.random.normal((batch_size, i, i, 1)) for i in self.scales]

        if self.phase == "TRANSITION":
            self.alpha.assign(tf.minimum(self.alpha + self.alpha_step, 1.0))
        elif self.phase == "STABLE":
            self.alpha.assign(1.0)
        else:
            raise NotImplementedError
        alpha_tmp = tf.tile(tf.expand_dims(tf.expand_dims(self.alpha, 0), 0), (batch_size, 1))

        for _ in range(1):
            # generator
            with tf.GradientTape() as g_tape:
                w = self.mapping(z)
                fake_images = self.G([const_input, w, alpha_tmp, noise])
                pred_fake = self.D([fake_images, alpha_tmp])
                g_loss = wasserstein_loss(real_labels, pred_fake)

                trainable_weights = (self.mapping.trainable_weights + self.G.trainable_weights)
                gradients = g_tape.gradient(g_loss, trainable_weights)
                self.g_optimizer.apply_gradients(zip(gradients, trainable_weights))

        for _ in range(2):
            # discriminator
            with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
                pred_fake = self.D([fake_images, alpha_tmp])
                pred_real = self.D([real_images, alpha_tmp])

                epsilon = tf.random.normal((batch_size, 1, 1, 1))
                interpolates = epsilon * real_images + (1 - epsilon) * fake_images
                gradient_tape.watch(interpolates)
                pred_interpolates = self.D([interpolates, alpha_tmp])

                loss_fake = wasserstein_loss(fake_labels, pred_fake)
                loss_real = wasserstein_loss(real_labels, pred_real)
                loss_interpolates = wasserstein_loss(fake_labels, pred_interpolates)

                # gradient penalty
                gradient_interpolates = gradient_tape.gradient(loss_interpolates, [interpolates])
                gradient_penalty = self.loss_weights["gradient_penalty"] * self.gradient_loss(gradient_interpolates)

                # drift loss
                all_pred = tf.concat([pred_fake, pred_real], axis=0)
                drift_loss = self.loss_weights["drift"] * tf.reduce_mean(tf.square(all_pred))

                d_loss = loss_fake + loss_real + gradient_penalty + drift_loss
                gradients = total_tape.gradient(d_loss, self.D.trainable_weights)
                self.d_optimizer.apply_gradients(zip(gradients, self.D.trainable_weights))

        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        self.d_loss_metric_fake.update_state(loss_fake)
        self.d_loss_metric_real.update_state(loss_real)
        self.d_loss_metric_penalty.update_state(gradient_penalty)
        self.d_loss_metric_drift.update_state(drift_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "loss_fake": self.d_loss_metric_fake.result(),
            "loss_real": self.d_loss_metric_real.result(),
            "gradient_penalty": self.d_loss_metric_penalty.result(),
            "drift_loss": self.d_loss_metric_drift.result(),
        }

    def call(self, inputs):
        return inputs