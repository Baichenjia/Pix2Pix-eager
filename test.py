import tensorflow as tf
import numpy as np
import os
from Pix2Pix.data_preprocess import load_image_test
from Pix2Pix.PixGenerator import Generator
from Pix2Pix.pix2pix import generated_image

# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers

# path
BASE_PATH = "/home/bai/Workspace/program/GANFamily/Pix2Pix"

# Model
print("load weights")
generator = Generator()
y = generator(tf.convert_to_tensor(np.random.random((1, 256, 256, 3)), tf.float32))
generator.load_weights(os.path.join(BASE_PATH, "weights/generator_134.h5"))
print("load done.")

# build test dataset
PATH = os.path.join(BASE_PATH, "data/facades/")
test_dataset = tf.data.Dataset.list_files(PATH + "test/*.jpg")
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)
print("dataset build done.")

# generate five image
for inp, tar in test_dataset.take(5):
    print("Generate image")
    generated_image(generator, inp, tar, t="test")


