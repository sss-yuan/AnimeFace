import math

import numpy as np
from PIL import Image

from models.Generator import Generator, Build_Mapping


def show():

    G_base = Generator(2, 8)
    M = Build_Mapping()

    G = G_base.get_model(7)
    G.load_weights("./logTest/test_G.h5")
    M.load_weights("./logTest/test_M.h5")

    np.random.seed(0)

    batch_size = 64
    z = np.random.normal(0, 1, size=(batch_size, 512))
    w = M.predict(z)

    const_inputs = np.ones((batch_size, 4, 4, 512))
    alpha = np.ones((batch_size, 1))
    noise = [np.random.normal(0, 1, size=(batch_size, 2 ** e, 2 ** e, 1)) for e in range(2, 9)]

    # fix_index = [0, 1]
    # for e in fix_index:
    #     for i in range(1, batch_size):
    #         noise[e][i] = noise[e][0]

    predict = G.predict([const_inputs, w, alpha, noise])
    images = np.clip((predict * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    w, h, _ = images[0].shape
    cols, rows = int(math.sqrt(batch_size)), int(math.sqrt(batch_size))
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(Image.fromarray(image), box=(i % rows * w, i // cols * h))
    grid.show()
    grid.save("show.png")


if __name__ == "__main__":
    show()