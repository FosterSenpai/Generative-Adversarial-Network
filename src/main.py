import signal
from pathlib import Path

import numpy as np
import tensorflow as tf

from gan_model import GAN

settings = {
    # Experiment
    "model_name": "cat_gan_resize_disc128_v1",
    "architecture_version": "dcgan_rgb_resize_disc128_v1",
    # Image preparation
    "image_size": 32,
    "channels": 3,
    "preprocessing": "bilinear_resize_stretch_rgb_minus1_plus1",  # Describe preproccesses done, will handle logic based on this later
    # Network
    "noise_dim": 100,
    "generator_leaky": False,
    "discriminator_leaky": True,
    # Training
    "epochs": 300,
    "batch_size": 256,
    "buffer_size": 60000,
    "learning_rate": 1e-4,
    # Saving and previews
    "checkpoint_dir": "training_checkpoints",
    "save_interval": 20,
    "examples_to_generate": 16,
}

gan = GAN(settings)
print(
    "Generator trainable parameters:",
    sum(int(np.prod(v.shape)) for v in gan.generator.trainable_variables),
)
print(
    "Discriminator trainable parameters:",
    sum(int(np.prod(v.shape)) for v in gan.discriminator.trainable_variables),
)

signal.signal(signal.SIGINT, gan.request_stop)  # Stop training on ctrl C

gan.load_image_directory(Path(r"C:\Users\foste\Downloads\cat image dataset"))
gan.save_training_preview()

# Check the model shapes before training.
noise = tf.random.normal([1, gan.noise_dim])
generated_image = gan.generator(noise, training=False)
prediction = gan.discriminator(generated_image, training=False)

print("Generator output:", generated_image.shape)
print("Discriminator output:", prediction.shape)

gan.train(gan.train_dataset)
gan.save_model()
