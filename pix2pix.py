import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from Pix2Pix.PixGenerator import Generator
from Pix2Pix.PixDiscriminator import Discriminator
from Pix2Pix.data_preprocess import load_image_train, load_image_test

# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers

# data
BASE_PATH = "/home/bai/Workspace/program/GANFamily/Pix2Pix"
PATH = os.path.join(BASE_PATH, "data/facades/")
BUFFER_SIZE = 400           # 训练图片共计400张
# batch设为1能在U-NET上产生较好结果，但不适合普通的Encoder-Decoder类型的Generator网络（见文章附录）
# 在普通的Generator网络下 BN 会将 BottleNeck层置为0.
BATCH_SIZE = 1              # Batch size的设置会影响到 BatchNormalization, 文章中说设置为1-10

# model
generator = Generator()
discriminator = Discriminator()
# optimizer
generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

"""
Discriminator loss:
The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

Generator loss:
It is a sigmoid cross entropy loss of the generated images and an array of ones.
The paper also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. 
This value was decided by the authors of the paper.
"""


def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_input, real_target],   真实的Input图像和target图像
       disc_generated_output = [real_image, generated_target]   真实的Input图像和生成的target图像
    """
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0
    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, l1_weight=100):
    """
        disc_generated_output: Generator生成的图像在Discriminator中的输出
        gen_output:  Generator生成的图像
        target:  target图像
        l1_weight: L1 损失的权重
        本来只需要第一项作为输入, 但文章中添加了生成图像与target图像的L1损失作为约束, 因此添加后两项输入
    """
    # GAN loss
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)
    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = tf.reduce_mean(gen_loss) + l1_weight * l1_loss
    return total_gen_loss


def generated_image(model, test_input, tar, t=0):
    """在当前参数下，给定输入，显示 Generator 生成的结果"""
    prediction = model(test_input)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0].numpy(), tar[0].numpy(), prediction[0].numpy()]
    title = ['Input image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # 原来输出范围是-1~1, 转为0~1显示
        plt.axis("off")
    plt.savefig(os.path.join(BASE_PATH, "generated_img", "img_"+str(t)+".jpg"))
    plt.show()


def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)                      # input -> generated_target
        disc_real_output = discriminator([input_image, target])  # [input, target] -> disc output
        disc_generated_output = discriminator([input_image, gen_output])  # [input, generated_target] -> disc output
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target)   # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(train_data, test_data, epochs):
    for epoch in range(epochs):
        print("-----\nEPOCH:", epoch)
        # train
        for input_image, target in train_data:
            gen_loss, disc_loss = train_step(input_image, target)
            print("Generator loss:", gen_loss.numpy(), ", Discriminator loss:", disc_loss.numpy())
        # generated and see the progress
        for inp, tar in test_data.take(1):
            generated_image(generator, inp, tar, t=epoch)

        # save checkpoint
        if (epoch + 1) % 15 == 0:
            generator.save_weights(os.path.join(BASE_PATH, "weights/generator_"+str(epoch)+".h5"))
            discriminator.save_weights(os.path.join(BASE_PATH, "weights/discriminator_"+str(epoch)+".h5"))


if __name__ == "__main__":
    # build train dataset
    train_dataset = tf.data.Dataset.list_files(PATH + "train/*.jpg")
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    # load_image_train 是已经定义好的函数，传入后每个图像都会经过该函数进行预处理
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # build test dataset
    test_dataset = tf.data.Dataset.list_files(PATH + "test/*.jpg")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(1)

    # train
    train(train_dataset, test_dataset, epochs=150)

