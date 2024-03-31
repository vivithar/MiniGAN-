import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Define the Generator architecture
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(784, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Define the Discriminator architecture
def build_discriminator(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Define function to generate and save images
def generate_images(generator, latent_dim, num_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir + "/generated_images.png")
    plt.show()

# Main function
if __name__ == "__main__":
    # Define hyperparameters
    latent_dim = 100
    epochs = 100
    batch_size = 64
    num_images_to_generate = 25
    output_dir = "generated_images"

    # Load and preprocess the dataset (e.g., MNIST)
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # Reduce the dataset by selecting the first 100 images
    train_images = train_images[:100]
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Build and compile the discriminator
    discriminator = build_discriminator(input_shape=(28, 28, 1))
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Build and compile the generator
    generator = build_generator(latent_dim)
    generator.compile(optimizer='adam', loss='binary_crossentropy')

    # Build and compile the GAN
    gan = models.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # Train the GAN
    for epoch in range(epochs):
        for _ in range(train_images.shape[0] // batch_size):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_images = generator.predict(noise)
            fake_images = fake_images.reshape(fake_images.shape[0], 28, 28, 1)  # Reshape fake_images to include the channel dimension
            real_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
            combined_images = np.concatenate([real_images, fake_images])
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)
            d_loss = discriminator.train_on_batch(combined_images, labels)
            misleading_targets = np.zeros((batch_size, 1))
            g_loss = gan.train_on_batch(noise, misleading_targets)
        print(f"Epoch {epoch + 1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Generate and save images
    generate_images(generator, latent_dim, num_images_to_generate, output_dir)
