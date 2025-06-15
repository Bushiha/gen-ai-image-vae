import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 2
decoder = tf.keras.models.load_model("models/decoder.h5")

# Sample random points in latent space
random_latent_vectors = np.random.normal(size=(10, latent_dim))
generated_images = decoder.predict(random_latent_vectors)

for i in range(10):
    plt.imshow(generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(f"images/generated_{i}.png")
    plt.show()
