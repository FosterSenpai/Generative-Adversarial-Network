import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

settings = {
    "model_name": "mnist_gan",
    "buffer_size": 60000,
    "batch_size": 256,
    "epochs": 5,
    "noise_dim": 100,
    "examples_to_generate": 16,
    "leaky": False,
    "save_interval": 15,
    "checkpoint_dir": "training_checkpoints",
}


class GAN:
    def __init__(self, settings: dict):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.train_dataset = None

        # Unpacking settings
        self.model_name = settings["model_name"]
        self.buffer_size = settings["buffer_size"]
        self.batch_size = settings["batch_size"]
        self.epochs = settings["epochs"]
        self.noise_dim = settings["noise_dim"]
        self.examples_to_generate = settings["examples_to_generate"]
        self.save_interval = settings["save_interval"]
        self.is_leaky = settings["leaky"]

        self.generator = self.create_generator(self.is_leaky)
        self.discriminator = self.create_discriminator(self.is_leaky)

        self.model_dir = Path(settings["checkpoint_dir"]) / self.model_name
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.image_dir = self.model_dir / "images"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Track completed epochs to resume training
        self.completed_epochs = tf.Variable(0, dtype=tf.int64, trainable=False)  # type: ignore

        # Tracking both models, both optimizers, training progress
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            completed_epochs=self.completed_epochs,
        )

        # keeping 3 most recent checkpoints
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.checkpoint_dir),
            max_to_keep=3,
        )

        # Fixed input for comparing generated images across epochs and restarts.
        self.preview_noise = tf.random.stateless_normal(
            [self.examples_to_generate, self.noise_dim],
            seed=[42, 0],  # type: ignore
        )

    def load_data(self, dir_path: Path | None):
        if dir_path:
            pass  # if local dir
        else:  # dl test
            (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

            train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")

            # Normalize imgs to [-1, 1]
            train_images = (train_images - 127.5) / 127.5

            # Batch and shuffle the data
            self.train_dataset = (
                tf.data.Dataset.from_tensor_slices(train_images)
                .shuffle(self.buffer_size)
                .batch(self.batch_size)
            )

    def handle_relu(self, leaky, model):
        if leaky:
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.ReLU())

    def create_generator(self, leaky: bool = False):
        model = tf.keras.Sequential()

        model.add(layers.Input(shape=(self.noise_dim,)))
        model.add(layers.Dense(7 * 7 * 256, use_bias=False))
        model.add(layers.BatchNormalization())
        self.handle_relu(leaky, model)

        model.add(layers.Reshape((7, 7, 256)))

        model.add(
            layers.Conv2DTranspose(
                128, (5, 5), strides=[1, 1], padding="same", use_bias=False
            )
        )
        model.add(layers.BatchNormalization())
        self.handle_relu(leaky, model)

        model.add(
            layers.Conv2DTranspose(
                64, (5, 5), strides=[2, 2], padding="same", use_bias=False
            )
        )
        model.add(layers.BatchNormalization())
        self.handle_relu(leaky, model)

        model.add(
            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=[2, 2],
                padding="same",
                use_bias=False,
                activation="tanh",
            )
        )

        return model

    def create_discriminator(self, leaky: bool = False):
        model = tf.keras.Sequential()
        model.add(
            layers.Conv2D(
                64,
                (5, 5),
                strides=(2, 2),
                padding="same",
                input_shape=[28, 28, 1],  # type: ignore
            )
        )

        self.handle_relu(leaky, model)
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        self.handle_relu(leaky, model)
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Calculate gradients for back prop
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        # Apply optimzers
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train(self, dataset):
        if dataset is None:
            raise ValueError("Load training data before calling train()")

        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored checkpoint: {latest_checkpoint}")

        start_epoch = int(self.completed_epochs.numpy())
        # Only save the untrained preview for a fresh run.
        if start_epoch == 0:
            self.generate_and_save_images(epoch=0)
        if start_epoch >= self.epochs:
            print(f"Already completed {start_epoch} epochs.")
            return

        for epoch in range(start_epoch, self.epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            self.completed_epochs.assign(epoch + 1)
            self.generate_and_save_images(epoch + 1)

            # Saving on interval
            if (epoch + 1) % self.save_interval == 0 or (epoch + 1) == self.epochs:
                checkpoint_path = self.checkpoint_manager.save()
                print(f"Checkpoint saved: {checkpoint_path}")

            elapsed = time.time() - start
            print(f"Time for epoch {epoch + 1}: {elapsed:.2f} sec")

    def generate_and_save_images(self, epoch):
        predictions = self.generator(self.preview_noise, training=False)

        # Convert from [-1, 1] to [0, 1]
        images = np.clip((predictions.numpy() + 1.0) / 2.0, 0.0, 1.0)

        count = self.examples_to_generate
        columns = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / columns))

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(columns * 2, rows * 2),
            squeeze=False,
        )

        for index, ax in enumerate(axes.flat):
            ax.axis("off")

            if index < count:
                ax.imshow(
                    images[index, :, :, 0],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )

        fig.tight_layout()

        image_path = self.image_dir / f"image_at_epoch_{epoch:04d}.png"
        fig.savefig(image_path)
        plt.close(fig)

    def save_model(self):
        # TODO: Should have settings file and output settings when saving model
        model_path = self.model_dir / f"{self.model_name}.keras"
        self.generator.save(str(model_path))
        print(f"Generator saved: {model_path}")

    def save_settings(self):
        config = {
            "model_name": self.model_name,
            "checkpoint_dir": str(self.model_dir.parent),
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "noise_dim": self.noise_dim,
            "examples_to_generate": self.examples_to_generate,
            "save_interval": self.save_interval,
            "leaky": self.is_leaky,
        }

        with open(self.model_dir / "settings.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        print(f"Settings saved: {self.model_dir / 'settings.json'}")

    def output_report(self):
        # TODO: Should find a way to output report on training like a notebook showing loss, imgs etc
        pass


if __name__ == "__main__":
    gan = GAN(settings)

    # Download and prepare MNIST.
    gan.load_data(dir_path=None)

    # Check the model shapes before training.
    noise = tf.random.normal([1, gan.noise_dim])
    generated_image = gan.generator(noise, training=False)
    prediction = gan.discriminator(generated_image, training=False)

    print("Generator output:", generated_image.shape)
    print("Discriminator output:", prediction.shape)

    gan.train(gan.train_dataset)
    gan.save_model()
    gan.save_settings()
