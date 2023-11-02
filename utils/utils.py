import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageEnhance


def get_img(img_path, resize=None, show_pic=False):
    img = _img_enhance(img_path, resize=resize)
    if show_pic:
        plt.subplot(111)
        plt.imshow(img)
        plt.show()
    img = img.astype(np.float64)
    img = img / 127.5
    img -= 1
    return img


def get_mask_img(img_path):
    if not os.path.exists(img_path):
        return np.zeros((256, 256))
    img = Image.open(img_path)
    if ".png" in img_path:
        img = img.convert("RGB")
    if img.size[0] == 512:
        img = img.resize((256, 256), Image.BICUBIC)
    img = np.array(img)
    img[img > 0] = 1
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = np.logical_or(r, g)
    img = np.logical_or(img, b)
    return img


def _rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a


def _img_enhance(img_path, bright=0.2, color=0.3, contrast=0.1, sharp=0.1, resize=None):

    img = Image.open(img_path)
    if ".png" in img_path:
        img = img.convert("RGB")
    if img.size[0] == 1024:
        img = img.resize((256, 256), Image.BICUBIC)
    if resize:
        img = img.resize(resize, Image.BICUBIC)

    # flip, 左右翻转
    # if _rand() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 调整亮度
    enh_bri = ImageEnhance.Brightness(img)
    img = enh_bri.enhance(factor=_rand(1 - bright, 1 + bright))

    # 调整图像的色彩平衡
    enh_color = ImageEnhance.Color(img)
    img = enh_color.enhance(factor=_rand(1 - color, 1 + color))

    # 调整图像的对比度
    enh_contrast = ImageEnhance.Contrast(img)
    img = enh_contrast.enhance(factor=_rand(1 - contrast, 1 + contrast))

    # 调整图像的锐化程度
    enh_sharp = ImageEnhance.Sharpness(img)
    img = enh_sharp.enhance(factor=_rand(1 - sharp, 1 + sharp))

    return np.array(img)


def get_data_generator_steps_per_epoch(batch_size):
    anime_face_path = './anime/'
    anime_face_files = os.listdir(anime_face_path)
    anime_face_num = len(anime_face_files)
    # anime_face_num = 10000

    return math.ceil(anime_face_num / batch_size)


def get_data_generator(batch_size, scale_log2):
    anime_face_path = './anime/'
    anime_face_files = os.listdir(anime_face_path)
    anime_face_num = len(anime_face_files)

    while True:
        anime_index_list = list(range(anime_face_num))
        np.random.shuffle(anime_index_list)

        index = 0
        while index < anime_face_num:
            batch_data = []
            for _ in range(batch_size):
                anime_img_path = anime_face_path + anime_face_files[anime_index_list[index]]
                anime_img = get_img(anime_img_path, resize=(2**scale_log2, 2**scale_log2))
                batch_data.append(anime_img)
                index += 1
                if index >= anime_face_num:
                    break
            batch_data = np.array(batch_data)
            # real_labels = np.ones_like(batch_data)
            # fake_labels = -np.ones_like(batch_data)
            yield batch_data


def get_canny_img(celebA_img_path, resize=None):
    img = cv2.imread(celebA_img_path)
    if resize:
        img = cv2.resize(img, resize)
    img = cv2.Canny(img, _rand(40, 90), _rand(130, 170))
    img = img.astype(np.float64)
    img = img / 127.5
    img -= 1
    return img


def get_celebA_canny_data(batch_size):
    celebA_path = './CelebAMask-HQ/CelebA-HQ-img/'
    celebA_num = 30000

    while True:
        celebA_index_list = list(range(celebA_num))
        np.random.shuffle(celebA_index_list)

        index = 0
        while index < celebA_num:
            batch_data = []
            for _ in range(batch_size):
                celebA_img_path = celebA_path + "%d.jpg" % index
                celebA_img = get_canny_img(celebA_img_path, resize=(128, 128))
                batch_data.append(celebA_img)
                index += 1
                if index >= celebA_num:
                    break
            batch_data = np.array(batch_data)
            yield batch_data


def get_celebA_canny_steps_per_epoch(batch_size):
    return math.ceil(30000 / batch_size)


def get_celebA_random_data(batch_size):
    celebA_path = './CelebAMask-HQ/CelebA-HQ-img/'
    celebA_num = 30000
    celebA_index_list = list(range(celebA_num))
    np.random.shuffle(celebA_index_list)
    random_index = celebA_index_list[:batch_size]
    data = []
    canny_data = []
    for i in random_index:
        celebA_img_path = celebA_path + "%d.jpg" % i
        celebA_img = get_img(celebA_img_path, resize=(128, 128))
        celebA_img_canny = get_canny_img(celebA_img_path, resize=(128, 128))
        data.append(celebA_img)
        canny_data.append(celebA_img_canny)
    return np.array(data), np.array(canny_data)


if __name__ == "__main__":
    test_path = "../CelebAMask-HQ/CelebA-HQ-img/"
    for ii in range(10):
        img_path = test_path + "%d.jpg" % ii
        get_img(img_path, show_pic=True)
