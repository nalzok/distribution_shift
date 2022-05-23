import h5py

import numpy as np
from scipy.stats import rankdata
import jax.numpy as jnp

DATA_PATH = 'data/3dshapes.h5'

# floor hue:    10 values linearly spaced in [0, 1]
# wall hue:     10 values linearly spaced in [0, 1]
# object hue:   10 values linearly spaced in [0, 1]
# scale:        8 values linearly spaced in [0, 1]
# shape:        4 values in [0, 1, 2, 3]
# orientation:  15 values linearly spaced in [-30, 30]
num_classes = [10, 10, 10, 8, 4, 15]


with h5py.File(DATA_PATH) as data:
    # Treat these as the population, not a sample
    images = np.array(data['images'], dtype=np.int8)
    labels = np.array(data['labels'])
    X_full = np.array(images, dtype=np.int8) # NHWC
    Y_full = rankdata(np.array(labels), 'dense', axis=0)


def random_sample(batch_size, weight, noise_scale):
    rng = np.random.default_rng(42)
    while True:
        idx = rng.choice(X_full.shape[0], batch_size, p=weight)
        X_batch, Y_batch = X_full[idx]/255., Y_full[idx]
        noise = rng.normal(0, noise_scale, X_batch.shape)
        X_batch = np.clip(X_batch + noise, 0, 1)
        yield jnp.array(X_batch), jnp.array(Y_batch)


def full_pass(batch_size, noise_scale, replication):
    rng = np.random.default_rng(42)
    for start in range(0, X_full.shape[0], batch_size):
        idx = np.arange(start, min(start + batch_size, X_full.shape[0]))
        X_batch, Y_batch = X_full[idx]/255., Y_full[idx]
        noise = rng.normal(0, noise_scale, (replication, *X_batch.shape))
        X_batch = np.clip(X_batch + noise, 0, 1)
        broadcaster = np.zeros((replication, *Y_batch.shape))
        Y_batch = broadcaster + Y_batch
        yield jnp.array(X_batch), jnp.array(Y_batch)


if __name__ == '__main__':
    weight = np.ones(480000) / 480000.
    batch_size = 4
    noise_scale = 0.1
    replication = 8

    train_loader = random_sample(batch_size, weight, noise_scale)
    X_batch, Y_batch = next(train_loader)
    assert X_batch.shape == (batch_size, 64, 64, 3)
    assert Y_batch.shape == (batch_size, 6)

    test_loader = full_pass(batch_size, noise_scale, replication)
    X_batch, Y_batch = next(test_loader)
    assert X_batch.shape == (replication, batch_size, 64, 64, 3)
    assert Y_batch.shape == (replication, batch_size, 6)