import time
from contextlib import redirect_stdout
import functools
import itertools

import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk
import optax

from dataloader import num_classes, Y_full, random_sample, full_pass


scale = Y_full[:, 3]/(num_classes[3] - 1)
shape = Y_full[:, 4]/(num_classes[4] - 1)
orientation = Y_full[:, 5]/(num_classes[5] - 1)

weights = {
    'uniform': np.ones(480000),
    'scale': scale,
    'shape': shape,
    'orientation': orientation,
    'scale+shape': scale + shape,
    'scale+orientation': scale + orientation,
    'shape+orientation': shape + orientation,
    'scale+shape+orientation': scale + shape + orientation,
    'scale*shape': scale * shape,
    'scale*orientation': scale * orientation,
    'shape*orientation': shape * orientation,
    'scale*shape*orientation': scale * shape * orientation,
}

models = [hk.nets.ResNet18, hk.nets.ResNet34, hk.nets.ResNet50,
        hk.nets.ResNet101, hk.nets.ResNet152, hk.nets.ResNet200]

def train():
    label_col = 3   # predict scale
    noise_scale = 0.01
    replication = jax.local_device_count()
    local_batch_size = 2048
    num_batches = 512
    learning_rate = 0.001

    experiment_id = int(time.time())
    logfile = f'logs/{experiment_id}_label{label_col}_noise{noise_scale.hex()}_rep{replication}_batch{local_batch_size}x{num_batches}_lr{learning_rate.hex()}.log'
    with open(logfile, 'x', buffering=1) as log:
        with redirect_stdout(log):

            for Model in models:
                def forward(X, is_training):
                    net = Model(num_classes[label_col], resnet_v2=True)
                    output = net(X, is_training)
                    return output

                forward_t = hk.transform_with_state(forward)
                forward_t = hk.without_apply_rng(forward_t)

                def learner_fn(X, y):
                    logits = forward(X, True)
                    labels = jax.nn.one_hot(y, num_classes[label_col])
                    return jnp.mean(optax.softmax_cross_entropy(logits, labels))

                learner_fn_t = hk.transform_with_state(learner_fn)
                learner_fn_t = hk.without_apply_rng(learner_fn_t)

                @functools.partial(jax.pmap, axis_name='batch')
                def train_step(params, state, opt_state, X, y):
                    (loss, state), grads = jax.value_and_grad(learner_fn_t.apply, has_aux=True)(params, state, X, y)
                    grads = jax.lax.pmean(grads, axis_name='batch')
                    loss = jax.lax.pmean(loss, axis_name='batch')
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return params, state, opt_state, loss
                
                optimizer = optax.adam(learning_rate)

                @jax.pmap
                def initial_state(rng, X, y):
                  params, state = learner_fn_t.init(rng, X, y)
                  opt_state = optimizer.init(params)
                  return params, state, opt_state

                @jax.pmap
                def eval_step(params, state, X, y):
                    logits, _ = forward_t.apply(params, state, X, False)
                    labels = jax.nn.one_hot(y, num_classes[label_col])
                    loss = optax.softmax_cross_entropy(logits, labels)
                    accuracy = jnp.argmax(logits, axis=-1) == y
                    return loss, accuracy

                for name, unnormalized in weights.items():
                    weight = unnormalized / np.sum(unnormalized)

                    outfile = f'results/{experiment_id}_label{label_col}_noise{noise_scale.hex()}_rep{replication}_batch{local_batch_size}x{num_batches}_lr{learning_rate.hex()}_{Model.__name__}_{name}.npz'
                    print('Unix epoch:', time.time())
                    print('Working on', outfile)

                    train_loader = random_sample(replication, local_batch_size, label_col, noise_scale, weight)
                    X, y = next(train_loader)
                    init_rng = jax.random.PRNGKey(42)
                    init_rng = jnp.broadcast_to(init_rng, (replication, *init_rng.shape))
                    params, state, opt_state = initial_state(init_rng, X, y)

                    # Training
                    for i, (X, y) in itertools.islice(enumerate(train_loader), num_batches):
                        params, state, opt_state, loss = train_step(params, state, opt_state, X, y)
                        # Note that loss is actually an array of shape [num_devices], with identical
                        # entries, because each device returns its copy of the loss, so we need
                        # to unreplicate it.
                        print(f'Train [{i+1}/{num_batches}]: {loss.mean()=}')

                    # Evaluating
                    pop_loss = np.empty((replication, 480000))
                    pop_accuracy = np.empty((replication, 480000), dtype=bool)
                    test_loader = full_pass(replication, local_batch_size, label_col, noise_scale)
                    for i, (X, y) in enumerate(test_loader):
                        batch_loss, batch_accuracy = eval_step(params, state, X, y)
                        start = i * local_batch_size
                        print(f'Test [{start}/480000]: {batch_loss.mean()=}, {batch_accuracy.mean()=}')
                        pop_loss[:, start:start+local_batch_size] = batch_loss
                        pop_accuracy[:, start:start+local_batch_size] = batch_accuracy
                    
                    np.savez_compressed(outfile, weight=weight, pop_loss=pop_loss, pop_accuracy=pop_accuracy)
                    print(f'Saved to {outfile}')


if __name__ == '__main__':
    train()
