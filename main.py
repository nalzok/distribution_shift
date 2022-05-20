import numpy as np
import jax
import jax.numpy as jnp

import optax
import equinox as eqx


DATA_PATH = 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


def dataloader(weight, batch_size, *, key):
    with np.load(DATA_PATH) as data:
        X = data['imgs']
        Y = data['latents_values']
        while True:
            idx = jax.random.choice(key, X.shape[0], (batch_size,), weight)
            yield X[idx], Y[idx]


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.nn.GRUCell
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jax.random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, input)
        return jax.nn.softmax(self.linear(out))


def main(
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    hidden_size=16,
    seed=42,
):
    loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed))
    weight = jnp.ones(737280) / 737280
    iter_data = dataloader(weight, batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    # pred_ys = jax.vmap(model)(xs)
    # num_correct = jnp.sum((pred_ys > 0.5) == ys)
    # final_accuracy = (num_correct / dataset_size).item()
    # print(f"final_accuracy={final_accuracy}")


if __name__ == '__main__':
    main()

