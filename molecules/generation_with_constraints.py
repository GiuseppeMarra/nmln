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
from molecules.utils import MoleculesHandler, MoleculesLogger
from molecules.data.molecules_data import molecules

np.random.seed(0)
tf.random.set_seed(0)
verbose = False


"""Setting up some parameters"""
# General NMLN parameters
flips = -1
num_chains = 10
burning_steps = 0
lr = 1e-4
p_noise = 0.01
MAX_NUM_ATOMS = 8
minibatch_size = 10
drop_rate = 0.
dataset = "molecules"
hierarchical = False
iterations = 200000
log_file = "generated_smiles.csv"

ks = [4]
hss = [[[150, tf.nn.relu],[75, tf.nn.relu]] ]
frags = [-1]



filenames, predicates = molecules(MAX_NUM_ATOMS)

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

"""Potential definition and instantiation"""
potentials = []
for i in range(len(ks)):
    if hierarchical:
        potentials.append(nmln.potentials.HierachicalPotential(n=MAX_NUM_ATOMS,
                                               k=ks[i],
                                               ontology=o, hidden_layers=hss[i]))
    else:
        potentials.append(nmln.potentials.NeuralMLPPotential(k=ks[i], ontology=o, hidden_layers=hss[i], num_sample_frags=frags[i]))
P = nmln.potentials.GlobalPotential(potentials)

"""Inference Method"""
# Constraints
c1 = nmln.potentials.OneOfNConstraint(list_predicates=[p.name for p in predicates if len(p.domains)==1],
                     ontology=o)
c2 = nmln.potentials.OneOfNConstraint(list_predicates=[p.name for p in predicates if len(p.domains)==2],
                      ontology=o,
                      symmetric_and_antireflexive=True,
                      alsoNone=True)
constraints = [c1, c2]
fs = [f for c in constraints for f in c.transformations()]
sampler = nmln.inference.GPUGibbsSamplerWithTransformations(potential=P,
                                                        transformations=fs,
                                                        num_examples= 2,
                                                        num_variables=o.linear_size(),
                                                        num_chains=num_chains//2)
adam = tf.keras.optimizers.Adam(lr)

"""Single update step"""
@tf.function
def train_step(pos_data, neg_data):

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


handler = MoleculesHandler(len(constants), o)
logger = MoleculesLogger(log_file, ontology=o, handler=handler)
"""Training routine"""
mols = []
for i in range(iterations):

    # Call to the single step
    pos_data = y[np.random.choice(len(y), minibatch_size)]
    pos_data = nmln.utils.generate_noise_coherent_with_transformations(fs, pos_data,p_noise)
    neg_data = sampler.sample()
    neg_data, positive_potentials, negative_potentials, loss = train_step(pos_data, neg_data)

    if i % 10 == 0:
        neg_data, positive_potentials, negative_potentials, loss = neg_data.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()
        print("Iter: %d, Pos: %.3f, Neg: %.3f, LL: %.3f " % (i, positive_potentials, negative_potentials, -loss))
        logger(neg_data)
