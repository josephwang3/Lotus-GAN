# Joseph Wang
# 12/26/2021
# Generates 64x64 pictures of lotuses using a TensorFlow DCGAN
# Based off of https://keras.io/examples/generative/dcgan_overriding_train_step/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import os

# prevent tensorflow logging output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# create data set from folder
# batch size is number of samples processed before model is updated
dataset = keras.preprocessing.image_dataset_from_directory(
    directory="lotus_new", label_mode=None, image_size=(64, 64), batch_size=32,
    shuffle=True
)

# rescale images to 0-1 range
dataset = dataset.map(lambda x: x / 255.0)

discriminator = keras.Sequential(
    [
        # size 64x64, 3 input channels (RGB)
        keras.Input(shape=(64, 64, 3)),

        # convolutional layers
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

discriminator.summary()

# noise
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),

        # create 8 x 8 image
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),

        # make image larger with convolutional tranpose (opposite of convolutional layers)
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),


        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ]
)

generator.summary()

# optimizers
opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)

# used to turn maximization problem into minimization problem
loss_fn = keras.losses.BinaryCrossentropy()

for epoch in range(8000):
    for idx, real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

        # generate fake image from random latent vectors
        fake = generator(random_latent_vectors)

        # save an image every once in a while in a file
        if idx % 7 == 0:
            img = keras.preprocessing.image.array_to_img(fake[0])
            img.save(f"generated_images/generated_img{epoch}_{idx}_.png")

        ### TRAIN DISCRIMINATOR, maximize y * log(Disc(x)) + (1 - y) * log(1 - Disc(Gen(z)))
        with tf.GradientTape() as disc_tape:
            # first term, send 1s to eliminate second term
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))

            # second term, send 0s to eliminate first term
            loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), discriminator(fake))
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
        
        # update discriminator
        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

        ### TRAIN GENERATOR, minimize log(1 - Disc(Gen(z)), or maximize log(Disc(Gen(z)))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)

            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)
        
        # update generator
        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(
            zip(grads, generator.trainable_weights)
        )