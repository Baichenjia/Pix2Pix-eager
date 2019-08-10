import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    """ 图像为256*512, 左侧和右侧分别为x和y, 需要切分 """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1]
    w = w // 2
    real_image = tf.cast(image[:, :w, :], tf.float32)     # 真实图, shape=(256,256,3)
    input_image = tf.cast(image[:, w:, :], tf.float32)    # 草图, shape=(256,256,3)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    """resize image to (height, width)"""
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    """random crop to (IMG_HEIGHT, IMG_WIDTH, 3)"""
    stacked_image = tf.stack([input_image, real_image], axis=0)   # (2, 256, 256, 3)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]   # input image and real image


def normalize(input_image, real_image):
    """ the images are normalized to (-1, 1) """
    input_image = (input_image / 127.5) - 1.
    real_image = (real_image / 127.5) - 1.
    return input_image, real_image


def random_jitter(input_image, real_image):
    """调用 resize, random_crop, normalize 方法 对图像进行预处理,
       先将图像扩大，随后进行 random_crop
    """
    input_image, real_image = resize(input_image, real_image, 286, 286)  # resize
    input_image, real_image = random_crop(input_image, real_image)
    # flip left right. 左右的镜面翻转
    if np.random.uniform() > 0.5:     # 括号中的括号代表只生成一个数字
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


def load_image_train(image_file):
    """load, jitter, and normalize"""
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_image_test(image_file):
    """测试数据不进行 random jitter 处理"""
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
