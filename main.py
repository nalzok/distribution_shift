import itertools

import h5py
import jax
import jax.numpy as jnp

import haiku as hk
import optax


DATA_PATH = 'data/3dshapes.h5'

# floor hue:    10 values linearly spaced in [0, 1]
# wall hue:     10 values linearly spaced in [0, 1]
# object hue:   10 values linearly spaced in [0, 1]
# scale:        8 values linearly spaced in [0, 1]
# shape:        4 values in [0, 1, 2, 3]
# orientation:  15 values linearly spaced in [-30, 30]
num_classes = [10, 10, 10, 8, 4, 15]

Model = hk.nets.ResNet18


def dataloader(weight, label_col, batch_size, *, key, fake_data = False):
    if fake_data:
        while True:
            yield jnp.zeros((batch_size, 64, 64, 3)), jnp.zeros(batch_size)

    with h5py.File(DATA_PATH) as data:
        X = jnp.array(data['images'])/255.  # NHWC
        Y = jnp.array(data['labels'][:, label_col])

        @jax.jit
        def next_batch(key):
            idx = jax.random.choice(key, X.shape[0], (batch_size,), p=weight)
            return key, X[idx], Y[idx]

        while True:
            key, X_batch, Y_batch = next_batch(key)
            yield X_batch, Y_batch


def train():
    Z_weight = jnp.ones(480000) / 480000
    label_col = 0
    batch_size = 32
    learning_rate = 0.001

    def forward(input, is_training):
        net = Model(num_classes[label_col], resnet_v2=True)
        output = net(input, is_training)
        return output

    def learner_fn(input, ground_truth):
        logits = forward(input, True)
        labels = jax.nn.one_hot(ground_truth, num_classes[label_col])
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    learner_fn_t = hk.transform_with_state(learner_fn)
    learner_fn_t = hk.without_apply_rng(learner_fn_t)

    @jax.jit
    def train_step(params, state, opt_state, X, y):
        (loss, state), grads = jax.value_and_grad(learner_fn_t.apply, has_aux=True)(params, state, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, state, opt_state, loss
    
    rng = jax.random.PRNGKey(42)
    loader_rng, rng = jax.random.split(rng)
    loader = dataloader(Z_weight, label_col, batch_size, key=loader_rng)

    X, y = next(loader)
    params, state = learner_fn_t.init(rng, X, y)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training
    for X, y in itertools.islice(loader, 50):
        params, state, opt_state, loss = train_step(params, state, opt_state, X, y)
        print(loss)

    forward_t = hk.transform_with_state(forward)
    forward_t = hk.without_apply_rng(forward_t)

    @jax.jit
    def eval_step(params, state, X):
        logits = forward_t.apply(params, state, X, False)
        return jnp.argmax(logits, axis=-1)

    # Evaluating
    for X, y in itertools.islice(loader, 50):
        predictions = eval_step(params, state, X)
        accuracy = jnp.mean(predictions == y)
        print(accuracy)


if __name__ == '__main__':
    train()
