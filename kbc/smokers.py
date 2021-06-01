import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import datasets
import numpy as np
from itertools import combinations,product,permutations
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
import mme


inter_sample_burn = 1


def check_for_duplicates(l):
    s = set()
    for i in l:
        if i in s:
            return True
        else:
            s.add(i)
    return False


def key(i,j):
    return min(i,j), max(i,j)

class Fragment():

    def __init__(self, i, j):

        self.i = min(i,j)
        self.j = max(i,j)
        self.rels_ij = []
        self.rels_ji = []

    def add_rel(self, h, t, r):
        if h<t:
            self.rels_ij.append(int(r))
        else:
            self.rels_ji.append(int(r))

    def exists_rel(self, h,t,r):
        if h<t:
            return r in self.rels_ij
        else:
            return r in self.rels_ji


    def get_interpretation(self, depth):
        i = np.zeros(depth).astype(np.int)
        i[np.concatenate((2*np.array(self.rels_ij), 2*np.array(self.rels_ji) + 1), axis=0).astype(np.int)]=1
        return i

    def key(self):
        return (self.i, self.j)


class NTPLikeScoringFunction():

    def __init__(self, marginal, ontology):
        self.marginal = marginal
        self.ontology = ontology

    def __call__(self, corrupted):
        scores = []
        for h,r,t in corrupted:
            size = self.ontology.predicates[r].domains[0].num_constants
            scores.append(self.marginal[0, self.ontology._predicate_range[r][0] + h * size + t])
        return scores





def main(dataset, num_samples, embedding_size, p_noise, lr, hidden_layers,flips):

    # Loading the dataset. Each split is a list of triples <h,r,t>
    constants, predicates, ground, train, valid, test = datasets.ntp_dataset(dataset)


    np.random.seed(0)
    tf.random.set_seed(0)


    d = mme.Domain("domains", num_constants=len(constants), constants=constants)
    p1 = mme.Predicate(name="friendOf", domains=[d,d]) #friends
    p2 = mme.Predicate(name="smokes", domains=[d])
    o = mme.Ontology(domains=[d], predicates=[p1,p2])



    num_variables = sum([len(constants)**len(p.domains) for n,p in o.predicates.items()])


    class KBCPotential(mme.potentials.Potential):

        def __init__(self, k, ontology, hidden_layers, num_constants, embedding_size):
            super(KBCPotential, self).__init__()
            self.embedding_size = embedding_size
            if self.embedding_size > 0:
                self.embeddings_entities = tf.Variable(
                    initial_value=tf.random.normal([num_constants, embedding_size]))
                self.embeddings_relations = tf.keras.layers.Dense(units=self.embedding_size, activation=None,
                                                                  use_bias=False)
            self.model = tf.keras.Sequential()
            for units, activation in hidden_layers:
                self.model.add(tf.keras.layers.Dense(units, activation=activation))
            self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))

            self.indices = np.array(list(permutations(range(num_constants), r=k)))
            groundings_hb_indices = []
            for i, (name, predicate) in enumerate(ontology.predicates.items()):
                predicate_range = ontology._predicate_range[name]
                size = predicate.domains[0].num_constants
                for j in range(k):
                    groundings_hb_indices.append(predicate_range[0] + size * self.indices[:, j:j + 1] + self.indices)

            self.groundings_hb_indices = np.concatenate(groundings_hb_indices, axis=1)

        def __call__(self, y, x=None):
            y = tf.gather(params=y, indices=self.groundings_hb_indices, axis=-1)
            e = tf.nn.embedding_lookup(self.embeddings_entities, self.indices)
            e = tf.reshape(e, [1] + list(self.indices.shape[:-1]) + [-1])
            if len(y.shape) > 3:  # if there is also the sampling dimension (i.e. when called by GibbsSampling)
                e = tf.tile(tf.expand_dims(e, axis=1), [1, y.shape[1], 1, 1])
            y = self.embeddings_relations(y)
            input = tf.concat((y, e), axis=-1)
            a = tf.squeeze(self.model(input))  # remove potential value 1 dimension
            return tf.reduce_sum(a, axis=-1)  # aggregate fragments

    class KBCPotentialNoConstantEmb(mme.potentials.Potential):

        def __init__(self, k, ontology, hidden_layers, num_constants):
            super(KBCPotentialNoConstantEmb, self).__init__()
            self.embedding_size = embedding_size
            self.model = tf.keras.Sequential()
            for units, activation in hidden_layers:
                self.model.add(tf.keras.layers.Dense(units, activation=activation))
            self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))

            self.indices = np.array(list(permutations(range(num_constants), r=k)))
            groundings_hb_indices = []
            for i, (name, predicate) in enumerate(ontology.predicates.items()):
                predicate_range = ontology._predicate_range[name]
                size = predicate.domains[0].num_constants
                for j in range(k):
                    groundings_hb_indices.append(predicate_range[0] + size*(len(predicate.domains)-1) * self.indices[:, j:j + 1] + self.indices)

            self.groundings_hb_indices = np.concatenate(groundings_hb_indices, axis=1)

        def __call__(self, y, x=None):
            y = tf.gather(params=y, indices=self.groundings_hb_indices, axis=-1)
            input = y
            a = tf.squeeze(self.model(input))  # remove potential value 1 dimension
            return tf.reduce_sum(a, axis=-1)  # aggregate fragments

    y = train.astype(np.float32)

    k = 3
    if embedding_size > 0:
        P = KBCPotential(k, o, hidden_layers, len(constants), embedding_size=embedding_size)
    else:
        P = KBCPotentialNoConstantEmb(k, o, hidden_layers, len(constants))
    P.beta = tf.ones(())
    sampler = mme.inference.GPUGibbsSampler(potential=P,
                                            num_examples=1,
                                            num_variables=num_variables,
                                            num_chains=num_samples,
                                            flips = flips if flips > 0 else None)
    adam = tf.keras.optimizers.Adam(lr)


    if embedding_size <= 0:
        sampler_test = mme.inference.GPUGibbsSampler(potential=P,
                                                num_examples=1,
                                                num_variables=num_variables,
                                                num_chains=num_samples,
                                                flips = flips if flips > 0 else None,
                                                initial_state=y,
                                                evidence=y,
                                                evidence_mask=y.astype(np.bool))


    def train_step(i):

        for _ in range(inter_sample_burn):
            neg_data = sampler.sample()

        noise = tf.random.uniform(shape=y.shape)
        pos_data = tf.where(noise > p_noise, y, 1 - y)

        with tf.GradientTape() as tape:

            positive_potentials = P(pos_data)
            positive_potentials = tf.reduce_mean(positive_potentials)
            negative_potentials = P(neg_data)
            negative_potentials = tf.reduce_mean(negative_potentials)
            loss = negative_potentials - positive_potentials

        gradients = tape.gradient(target=loss,sources=P.variables)
        grads_vars = zip(gradients, P.variables)
        adam.apply_gradients(grads_vars)

        if embedding_size<=0:
            test = sampler_test.sample()
            return test.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()

        return neg_data.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()


    #Evaluation Staff
    #Fragment_indices_to_row_in_marginals

    MARG = np.zeros(y.shape)

    test_res = 0
    with np.printoptions(precision=2, suppress=True):
        for i in range(0, 200000):

            samples, positive_potentials, negative_potentials, loss = train_step(i)
            MARG += np.sum(samples, axis=1)
            M = MARG / ((i + 1) * num_samples)
            for n,p in o.predicates.items():
                print(np.reshape(M[:, o._predicate_range[n][0]:o._predicate_range[n][1]], [d.num_constants for d in p.domains]))




if __name__=="__main__":

    RES = []

    TRIALS = [0]
    DATASETS = ["smokers"]
    NUM_SAMPLES = (10,)
    EMBEDDING_SIZES = (0,)
    P_NOISES = (0.,)
    LRS = [1e-2]
    HIDDEN_LAYERS = [((30, tf.nn.sigmoid),)]
    FLIPS = [-1]


    for A in product(TRIALS, FLIPS,DATASETS, NUM_SAMPLES, EMBEDDING_SIZES, P_NOISES, LRS, HIDDEN_LAYERS):
        _, flips, dataset, num_samples,embedding_size,p_noise,lr, hidden_layers = A
        r = main(dataset,num_samples,embedding_size,p_noise,lr,hidden_layers,flips)
        RES.append((A, r))
        for i in RES: print(i)
        with open("res_smokers", "a") as f:
            for i in RES:
                f.write(str(i))
                f.write("\n")






