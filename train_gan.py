import os
import tensorflow as tf
import keras
import numpy as np
from models.GAN import GAN
from utils.utils import get_data_generator, get_data_generator_steps_per_epoch


def train():
    gan = GAN()
    scales_log2 = [2, 3, 4, 5, 6, 7, 8]
    batch_sizes = [16, 16, 12, 12, 12, 12, 8]
    learning_rate = 1e-3

    global_step = 1
    for i, scale_log2 in enumerate(scales_log2):
        batch_size = batch_sizes[i]

        steps_per_epoch = get_data_generator_steps_per_epoch(batch_size)
        data_generator = get_data_generator(batch_size, scale_log2)

        gan.compile(
            g_optimizer=keras.optimizers.adam_v2.Adam(learning_rate=learning_rate),
            d_optimizer=keras.optimizers.adam_v2.Adam(learning_rate=learning_rate),
            scale_log2=scale_log2,
            loss_weights={"gradient_penalty": 10, "drift": 0.001},
        )
        G_path = f"./logs/{2**scale_log2}x{2**scale_log2}/G/init.h5"
        D_path = f"./logs/{2 ** scale_log2}x{2 ** scale_log2}/D/init.h5"
        M_path = f"./logs/{2 ** scale_log2}x{2 ** scale_log2}/M/init.h5"
        if os.path.exists(G_path):
            gan.G.load_weights(G_path)
            print(f"G load weights from {G_path}")
        if os.path.exists(D_path):
            gan.D.load_weights(D_path)
            print(f"D load weights from {D_path}")
        if os.path.exists(M_path):
            gan.mapping.load_weights(M_path)
            print(f"mapping load weights from {M_path}")
        if os.path.exists(G_path) and os.path.exists(D_path) and os.path.exists(M_path):
            continue

        gan.set_phase("TRANSITION")
        print("TRANSITION")
        gan.fit(
            data_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=4,
        )

        gan.set_phase("STABLE")
        print("STABLE")
        tmp_result = gan.fit(
            data_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=4,
        )
        d_loss = np.mean(tmp_result.history["d_loss"])
        g_loss = np.mean(tmp_result.history["g_loss"])
        postfix = f"{global_step:04d}_d_loss_{d_loss:.3f}-g_loss_{g_loss:.3f}"
        global_step += 1
        gan.my_save(postfix)
        gan.gen_sample_img(scale_log2, postfix)

        threshold = 0.35 if i != 0 else 1.1
        while abs(d_loss) > 2 * threshold or abs(g_loss+1) > threshold:
            tmp_result = gan.fit(
                data_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=2,
            )
            d_loss = np.mean(tmp_result.history["d_loss"])
            g_loss = np.mean(tmp_result.history["g_loss"])
            postfix = f"{global_step:04d}_d_loss_{d_loss:.3f}-g_loss_{g_loss:.3f}"
            global_step += 1
            gan.my_save(postfix)
            gan.gen_sample_img(scale_log2, postfix)


if __name__ == "__main__":
    train()