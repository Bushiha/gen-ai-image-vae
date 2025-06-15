import tensorflow as tf
from model import vae

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

vae.fit(x_train, epochs=30, batch_size=128)
vae.encoder.save("models/encoder.h5")
vae.decoder.save("models/decoder.h5")
