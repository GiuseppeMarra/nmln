import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import datasets
import numpy as np
import nmln
from generation.molecules_utils import MoleculesHandler
from rdkit import Chem

np.random.seed(0)
tf.random.set_seed(0)

"""Setting up some parameters"""
# General NMLN parameters
flips = -50
num_chains = 5
num_samples_expected_value = num_chains
burning_steps = 0
lr = 0.001
p_noise = 0
MAX_NUM_ATOMS = 8
minibatch_size = 50
drop_rate = 0.
verbose = False

# This lists contains parameters for creating the potential functions
ks = [4]
hss = [[[150, tf.nn.relu], [50, tf.nn.relu]]]
frags=[-1]

filenames, predicates = datasets.molecules(MAX_NUM_ATOMS)

"""Knowledge Representation"""
constants = [str(i) for i in range(MAX_NUM_ATOMS)]
d = nmln.Domain(name="atoms", constants=constants, num_constants=len(constants))
predicates = [nmln.Predicate(p, domains=[d for _ in range(a)]) for p, a in predicates.items()]
o = nmln.Ontology(domains=[d], predicates=predicates)


"""Data Loading. FOL descriptions are serialized into flat tensors"""
# The creation of the Y from a file should be highly improved
Y = []
for filename in filenames:
    y = o.file_to_linearState(filename)
    Y.append(y)
print(len(Y))
Y = np.stack(Y, axis=0).astype(np.float32)
y = Y



potentials = []
for i in range(len(ks)):
    potentials.append(nmln.potentials.NeuralMLPPotential(k=ks[i], ontology=o, hidden_layers=hss[i], num_sample_frags=frags[i]))
P = nmln.potentials.GlobalPotential(potentials)

"""Inference Method"""
sampler = nmln.inference.GPUGibbsSampler(potential=P,
                                        num_examples=minibatch_size // num_chains,
                                        num_variables=o.linear_size(),
                                        num_chains=num_chains,
                                        flips=flips)
adam = tf.keras.optimizers.Adam(lr)

"""Single update step"""

@tf.function
def train_step(neg_data):

    noise = tf.random.uniform(shape=y.shape)
    pos_data = tf.where(noise > p_noise, y, 1 - y)
    pos_data = tf.random.shuffle(pos_data)[:minibatch_size]

    with tf.GradientTape() as tape:
        positive_potentials = P(pos_data)
        positive_potentials = tf.reduce_mean(positive_potentials)
        negative_potentials = P(neg_data)
        negative_potentials = tf.reduce_mean(negative_potentials)
        loss = negative_potentials - positive_potentials

    gradients = tape.gradient(target=loss, sources=P.variables)
    grads_vars = zip(gradients, P.variables)
    adam.apply_gradients(grads_vars)

    return neg_data, positive_potentials, negative_potentials, loss


# Removing RDKit logs for the moment
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


handler = MoleculesHandler(len(constants), o)
"""Training routine"""
mols = []
for i in range(2000000):

    # Call to the single step
    neg_data = sampler.sample()
    neg_data, positive_potentials, negative_potentials, loss = train_step(neg_data)
    neg_data, positive_potentials, negative_potentials, loss = neg_data.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()
    print(positive_potentials, negative_potentials, -loss)


