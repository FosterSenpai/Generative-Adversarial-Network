import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

""" SETTINGS LOOKS LIKE THIS
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
"""


class GAN:
    def __init__(self, settings: dict):
        # Unpacking settings
        self.model_name = settings["model_name"]
        self.learning_rate = settings["learning_rate"]
        self.buffer_size = settings["buffer_size"]
        self.batch_size = settings["batch_size"]
        self.epochs = settings["epochs"]
        self.noise_dim = settings["noise_dim"]
        self.examples_to_generate = settings["examples_to_generate"]
        self.save_interval = settings["save_interval"]
        self.is_leaky = settings["leaky"]
        self.image_size = settings["image_size"]
        self.channels = settings["channels"]
        self.architecture_version = settings["architecture_version"]
        self.preprocessing = settings[
            "preprocessing"
        ]  # Could apply different preprocessing based off this string

        # Settings checks
        if (self.image_size, self.channels) != (32, 3):
            raise ValueError("The current architecture requires 32×32 RGB images.")
        if self.architecture_version != "dcgan_rgb_v1":
            raise ValueError("Unsupported architecture version.")
        if self.preprocessing != "bilinear_resize_stretch_rgb_minus1_plus1":
            raise ValueError("Unsupported preprocessing.")

        self.generator = self.create_generator(self.is_leaky)
        self.discriminator = self.create_discriminator(self.is_leaky)

        self.generator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.train_dataset = None
        self.dataset_source = None

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

    def load_data(self, images, *, source: str):
        """Load an image array dataset
        Args:
            images: The images to load.
            source (str): A string describing the dataset source for documentation.
        """
        images = np.asarray(images)

        # Safety checks
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB images shapes (N, height, width, 3).Got {images.shape}."
            )
        if len(images) == 0:
            raise ValueError("The dataset containes no images.")

        def prepare_image(image):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (32, 32))
            return (image - 127.5) / 127.5

        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(images)
            .shuffle(self.buffer_size)
            .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.dataset_source = source

    def load_image_directory(self, dir_path: Path | str):
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise ValueError(f"Image directory does not exist: {dir_path}")

        dataset = tf.keras.utils.image_dataset_from_directory(
            str(dir_path),
            labels=None,
            color_mode="rgb",
            image_size=(32, 32),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.train_dataset = dataset.map(
            lambda images: (images - 127.5) / 127.5,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        self.dataset_source = str(dir_path.resolve())

    def handle_relu(self, leaky, model):
        if leaky:
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.ReLU())

    def create_generator(self, leaky: bool = False):
        model = tf.keras.Sequential()

        model.add(layers.Input(shape=(self.noise_dim,)))
        model.add(layers.Dense(8 * 8 * 256, use_bias=False))
        model.add(layers.BatchNormalization())
        self.handle_relu(leaky, model)

        model.add(layers.Reshape((8, 8, 256)))

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
                3,
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

        model.add(layers.Input(shape=(32, 32, 3)))
        model.add(
            layers.Conv2D(
                64,
                (5, 5),
                strides=(2, 2),
                padding="same",
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

    def reconcile_loss_history(self, completed_epoch):
        """Trim any rows in history newer than the restored checkpoint to avoid overlap"""
        history_path = self.model_dir / "loss_history.csv"

        if not history_path.exists() or history_path.stat().st_size == 0:  # No history
            return

        # Read rows
        with history_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames

            if not fieldnames or "epoch" not in fieldnames:
                raise ValueError("Loss history is missing its epoch column.")

            rows = list(reader)

        # Keep before current
        keeping = {}
        for row in rows:
            epoch = int(row["epoch"])

            if 1 <= epoch <= completed_epoch:
                keeping[epoch] = row

        cleaned_rows = [keeping[epoch] for epoch in sorted(keeping)]

        if cleaned_rows == rows:
            return

        # Write into temp file then replace
        temp_path = history_path.with_suffix(".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        temp_path.replace(history_path)

    def reconcile_preview_images(self, completed_epoch):
        """Remove genereated previews newer than the restored checkpoint"""
        removed = 0

        for image_path in self.image_dir.glob("image_at_epoch_*.png"):
            epoch_text = image_path.stem.removeprefix("image_at_epoch_")

            if not epoch_text.isdecimal():
                continue

            if int(epoch_text) > completed_epoch:
                image_path.unlink()
                removed += 1

        if removed:
            print(f"Removed {removed} previews newer than epoch {completed_epoch}.")

    def train(self, dataset):
        if dataset is None:
            raise ValueError("Load training data before calling train()")

        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.validate_resume_settings()
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored checkpoint: {latest_checkpoint}")
        self.save_settings()

        start_epoch = int(self.completed_epochs.numpy())

        # Reconcile recorded stats and images to not be newer than restored checkpoint
        self.reconcile_loss_history(start_epoch)
        self.reconcile_preview_images(start_epoch)

        # Only save the untrained preview for a fresh run.
        if start_epoch == 0:
            self.generate_and_save_images(epoch=0)
        if start_epoch >= self.epochs:
            print(f"Already completed {start_epoch} epochs.")
            return

        for epoch in range(start_epoch, self.epochs):
            start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            gen_average = tf.keras.metrics.Mean()
            disc_average = tf.keras.metrics.Mean()

            progress = tf.keras.utils.Progbar(
                target=len(dataset),
                unit_name="batch",
            )

            for batch_index, image_batch in enumerate(dataset):
                gen_loss, disc_loss = self.train_step(image_batch)

                # Weight by batch size because the last batch may be smaller.
                batch_count = tf.shape(image_batch)[0]
                gen_average.update_state(gen_loss, sample_weight=batch_count)
                disc_average.update_state(disc_loss, sample_weight=batch_count)

                progress.update(
                    batch_index + 1,
                    values=[
                        ("gen_loss", float(gen_loss.numpy())),
                        ("disc_loss", float(disc_loss.numpy())),
                    ],
                )

            # Saving history after each epoch
            history_path = self.model_dir / "loss_history.csv"
            write_header = not history_path.exists() or history_path.stat().st_size == 0

            with history_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                if write_header:
                    writer.writerow(["epoch", "gen_loss", "disc_loss"])

                writer.writerow(
                    [
                        epoch + 1,
                        float(gen_average.result().numpy()),
                        float(disc_average.result().numpy()),
                    ]
                )

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
                ax.imshow(images[index])

        fig.tight_layout()

        image_path = self.image_dir / f"image_at_epoch_{epoch:04d}.png"
        fig.savefig(image_path)
        plt.close(fig)

    def save_model(self):
        model_path = self.model_dir / f"{self.model_name}.keras"
        self.generator.save(str(model_path))
        print(f"Generator saved: {model_path}")

    def get_config(self):
        config = {
            # Experiment
            "model_name": self.model_name,
            "architecture_version": self.architecture_version,
            # Image preparation
            "dataset_source": self.dataset_source,
            "image_size": self.image_size,
            "channels": self.channels,
            "preprocessing": self.preprocessing,
            # Network
            "noise_dim": self.noise_dim,
            "leaky": self.is_leaky,
            # Training
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "learning_rate": self.learning_rate,
            # Saving and previews
            "checkpoint_dir": str(self.model_dir.parent),
            "save_interval": self.save_interval,
            "examples_to_generate": self.examples_to_generate,
        }
        return config

    def validate_resume_settings(self):
        """Checks if config for model being resumed matches set config to avoid clashes."""

        # Get current and saved settings
        settings_path = self.model_dir / "settings.json"
        if (
            not settings_path.exists()
        ):  # TODO: Maybe add a force to override this if no settings.
            raise ValueError(
                "Checkpoint found, but settings.json is missing."
                "Cannot verify that run is compatible."
            )
        with settings_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        current = self.get_config

        # These must stay consistent across experments CANNOT BE CHANGED
        required_keys = (
            "architecture_version",
            "dataset_source",
            "image_size",
            "channels",
            "preprocessing",
            "noise_dim",
            "leaky",
            "learning_rate",
        )

        # Checking for any missing keys in saved settings
        missing = [key for key in required_keys if key not in saved]
        if missing:
            raise ValueError(
                "Saved settings are missing these fields: " + ", ".join(missing)
            )

        # Checking for differences
        differences = []
        for key in required_keys:
            saved_value = saved[key]
            current_value = current[key]

            if saved_value != current_value:
                differences.append(
                    f"{key}: saved={saved_value}, current={current_value}"
                )

        if differences:
            details = "\n".join(differences)
            raise ValueError(
                f"Settings do not match the saved experiment:\n{details}\n"
                "Use a new model name for a different experiment."
            )

    def save_settings(self):
        config = self.get_config()
        with open(self.model_dir / "settings.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Settings saved: {self.model_dir / 'settings.json'}")

    def save_training_preview(self):
        if self.train_dataset is None:
            raise ValueError("Load training data before saving a preview.")

        # Collect individual images, even if they span multiple batches.
        images = list(
            self.train_dataset.unbatch()
            .take(self.examples_to_generate)
            .as_numpy_iterator()
        )

        if not images:
            raise ValueError("The training dataset contains no images.")

        # Convert normalized pixels back to [0, 1] for display.
        images = np.clip((np.stack(images) + 1.0) / 2.0, 0.0, 1.0)

        count = len(images)
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
                ax.imshow(images[index], interpolation="nearest")

        fig.tight_layout()

        # Keep the reference separate from generated epoch images.
        preview_path = self.model_dir / "real_training_preview.png"
        fig.savefig(preview_path)
        plt.close(fig)

        print(f"Training preview saved: {preview_path}")

    def output_report(self):
        # TODO: Should find a way to output report on training like a notebook showing loss, imgs etc
        pass


# Example test, training on dataset of cats
if __name__ == "__main__":
    settings = {
        # Experiment
        "model_name": "cat_gan",
        "architecture_version": "dcgan_rgb_v1",
        # Image preparation
        "image_size": 32,
        "channels": 3,
        "preprocessing": "bilinear_resize_stretch_rgb_minus1_plus1",  # Describe preproccesses done, will handle logic based on this later
        # Network
        "noise_dim": 100,
        "leaky": False,
        # Training
        "epochs": 500,
        "batch_size": 256,
        "buffer_size": 60000,
        "learning_rate": 1e-4,
        # Saving and previews
        "checkpoint_dir": "training_checkpoints",
        "save_interval": 15,
        "examples_to_generate": 16,
    }

    gan = GAN(settings)

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
