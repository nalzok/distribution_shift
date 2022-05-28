import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def main(divergence_measure):
    weights = {}
    pop_losses = {}
    pop_accuracies = {}
    models = set()

    uniform = np.ones(480000)/480000
    timestamp = os.environ['TIMESTAMP']

    for filename in glob.glob(f'results/{timestamp}_*.npz'):
        _, label, noise, rep, batch, lr, model, setting = filename.split('_')
        label_col = int(label[5:])
        noise_scale = float.fromhex(noise[5:])
        replication = int(rep[3:])
        local_batch_size, num_batches = batch[5:].split('x')
        local_batch_size = int(local_batch_size)
        num_batches = int(num_batches)
        learning_rate = float.fromhex(lr[2:])
        setting = setting[:-4]

        models.add(model)

        with np.load(filename) as data:
            weights[setting] = data['weight']
            pop_losses[(setting, model)] = np.mean(data['pop_loss'], axis=0)
            pop_accuracies[(setting, model)] = np.mean(data['pop_accuracy'], axis=0)

    for model in models:
        unbalancednesses = []
        divergences = []
        expected_losses = []
        expected_accuracies = []
        for train_setting in weights.keys():
            q = weights[train_setting]
            unbalancedness = divergence_measure(q, uniform)
            for test_setting in weights.keys():
                p = weights[test_setting]
                unbalancednesses.append(unbalancedness)
                divergences.append(divergence_measure(p, q))
                pop_loss = pop_losses[(train_setting, model)]
                expected_losses.append(np.sum(p * pop_loss))
                pop_accuracy = pop_accuracies[(train_setting, model)]
                expected_accuracies.append(np.sum(p * pop_accuracy))

        plt.scatter(unbalancednesses, divergences, c=expected_losses,
                label='Losses', cmap='magma')
        plt.xlabel('Unbalancedness')
        plt.ylabel('Test-train Divergence')
        plt.legend()
        plt.colorbar()
        plt.savefig(f'plots/experiment_qys_{model}_{label_col}_losses.png')
        plt.clf()

        plt.scatter(unbalancednesses, divergences, c=expected_accuracies,
                label='Accuracies', cmap='magma_r')
        plt.xlabel('Unbalancedness')
        plt.ylabel('Test-train Divergence')
        plt.legend()
        plt.colorbar()
        plt.savefig(f'plots/experiment_qys_{model}_{label_col}_accuracies.png')
        plt.clf()



def kl_divergence(p, q):
    """
    Calculate the KL divergence from Q to P, i.e. KL(P||Q).
    In other words, Q is the reference distribution.
    """
    return np.sum(p * np.log(p/q))


def total_variation_distance(p, q):
    return np.sum(np.abs(p-q))/2



if __name__ == '__main__':
    main(total_variation_distance)
