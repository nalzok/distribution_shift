import time
from contextlib import redirect_stdout
import itertools

import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk
import optax

from dataloader import num_classes, Y_full, random_sample, full_pass


scale = Y_full[:, 3]/num_classes[3]
shape = Y_full[:, 4]/num_classes[4]
orientation = Y_full[:, 5]/num_classes[5]

weights = {
    'uniform': np.ones(480000),
    'scale': scale,
    'shape': shape,
    'orientation': orientation,
    'scale*shape': scale * shape,
    'scale*orientation': scale * orientation,
    'shape*orientation': shape * orientation,
    'scale*shape*orientation': scale * shape * orientation,
}

models = [hk.nets.ResNet200, hk.nets.ResNet152, hk.nets.ResNet101,
        hk.nets.ResNet50, hk.nets.ResNet34, hk.nets.ResNet18]

def train():
    label_col = 4   # predict shape
    batch_size = 2048
    num_batches = 1024
    noise_scale = 0.01
    learning_rate = 0.001
    test_replication = jax.device_count()

    experiment_id = int(time.time())
    logfile = f'logs/{experiment_id}_label{label_col}_noise{noise_scale.hex()}_batch{batch_size}x{num_batches}_lr{learning_rate.hex()}_rep{test_replication}.log'
    with open(logfile, 'x') as log:
        with redirect_stdout(log):

            for Model in models:
                def forward(X, is_training):
                    net = Model(num_classes[label_col], resnet_v2=True)
                    output = net(X, is_training)
                    return output

                def learner_fn(X, y):
                    logits = forward(X, True)
                    labels = jax.nn.one_hot(y[:, label_col], num_classes[label_col])
                    return jnp.mean(optax.softmax_cross_entropy(logits, labels))

                learner_fn_t = hk.transform_with_state(learner_fn)
                learner_fn_t = hk.without_apply_rng(learner_fn_t)

                @jax.jit
                def train_step(params, state, opt_state, X, y):
                    (loss, state), grads = jax.value_and_grad(learner_fn_t.apply, has_aux=True)(params, state, X, y)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return params, state, opt_state, loss
                
                optimizer = optax.adam(learning_rate)

                @jax.jit
                def initial_state(rng, X, y):
                  params, state = learner_fn_t.init(rng, X, y)
                  opt_state = optimizer.init(params)
                  return params, state, opt_state

                for name, unnormalized in weights.items():
                    weight = unnormalized / np.sum(unnormalized)

                    outfile = f'results/{experiment_id}_label{label_col}_noise{noise_scale.hex()}_batch{batch_size}x{num_batches}_lr{learning_rate.hex()}_rep{test_replication}_{Model.__name__}_{name}.npz'
                    print('Unix epoch:', time.time())
                    print('Working on', outfile)

                    train_loader = random_sample(batch_size, weight, noise_scale)
                    X, y = next(train_loader)
                    init_rng = jax.random.PRNGKey(42)
                    params, state, opt_state = initial_state(init_rng, X, y)

                    # Training
                    for i, (X, y) in itertools.islice(enumerate(train_loader), num_batches):
                        params, state, opt_state, loss = train_step(params, state, opt_state, X, y)
                        loss = jnp.asarray(loss).item()
                        print(f'Train [{i+1}/{num_batches}]: {loss=}')

                    forward_t = hk.transform_with_state(forward)
                    forward_t = hk.without_apply_rng(forward_t)

                    @jax.jit
                    @jax.pmap
                    def eval_step(params, state, X, y):
                        logits, _ = forward_t.apply(params, state, X, False)
                        labels = jax.nn.one_hot(y[:, label_col], num_classes[label_col])
                        loss = optax.softmax_cross_entropy(logits, labels)
                        accuracy = jnp.argmax(logits, axis=-1) == y
                        return loss, accuracy

                    # Evaluating
                    pop_loss = np.empty((test_replication, 480000))
                    pop_accuracy = np.empty((test_replication, 480000), dtype=bool)
                    test_loader = full_pass(batch_size, noise_scale, test_replication)
                    for i, (X, y) in enumerate(test_loader):
                        batch_loss, batch_accuracy = jnp.asarray(eval_step(params, state, X, y))
                        start = i * batch_size
                        print(f'Test [{start}/480000]: {np.mean(batch_loss)=}, {np.mean(batch_accuracy)=}')
                        pop_loss[:, start:start+batch_size] = batch_loss
                        pop_accuracy[:, start:start+batch_size] = batch_accuracy
                    
                    np.savez_compressed(outfile, weight=weight, pop_loss=pop_loss, pop_accuracy=pop_accuracy)
                    print(f'Saved to {outfile}')


if __name__ == '__main__':
    train()
