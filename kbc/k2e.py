import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from collections import OrderedDict
from itertools import combinations,product
import tensorflow as tf
from kbc_utils import Ranking
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
import nmln
from nmln.utils import ntp_dataset_triple




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

# This class represent k=2 fragments and provides a utility to translate the fragment into its Herbrand Base
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


# Scoring function from the NTP paper adapted to out setting
class NTPLikeScoringFunction():

    def __init__(self, marginal, d):
        self.marginal = marginal
        self.d = d

    def __call__(self, corrupted):
        scores = []
        for h,r,t in corrupted:
            if h==t:
                scores.append(0.)#NMLN-T does not support reflexive atoms. Automatically puts a 0 score
            else:
                scores.append(self.marginal[self.d[key(h, t)], 2 * r if h < t else 2 * r + 1])
        return scores



# This class parallelize the evaluations on all the fragments and their anonymization. It is ad-hoc for this setting
class KBCTractablePotential(nmln.potentials.Potential):

    def __init__(self, hidden_layers, num_constants, embedding_size, num_variables):
        super(KBCTractablePotential, self).__init__()
        self.embedding_size = embedding_size
        if self.embedding_size > 0:
            self.embeddings_entities = tf.Variable(initial_value=tf.random.normal([num_constants, embedding_size]))
            self.embeddings_relations = tf.keras.layers.Dense(units=self.embedding_size, activation=None,
                                                              use_bias=False)
        self.model = tf.keras.Sequential()
        for units, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(units, activation=activation))
        self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))
        self.permutation = [a + 1 if a % 2 == 0 else a - 1 for a in range(num_variables)]

    def __call__(self, y, x=None):
        if x is None:
            raise Exception("KBCTractablePotential needs x when called")

        y_permuted = tf.gather(params=y, indices=self.permutation, axis=-1)
        e = tf.nn.embedding_lookup(self.embeddings_entities, x)
        e_permuted = tf.reshape(tf.gather(params=e, indices=[1, 0], axis=-2), list(x.shape[:-1]) + [-1])
        e = tf.reshape(e, list(x.shape[:-1]) + [-1])
        e = tf.stack((e, e_permuted), axis=0)
        y = tf.stack((y, y_permuted), axis=0)
        y = self.embeddings_relations(y)
        input = tf.concat((y, e), axis=-1)

        a = tf.squeeze(self.model(input))
        return tf.reduce_sum(a, axis=0)


def main(dataset, num_samples, embedding_size, p_noise, lr, hidden_layers):

    # Loading the dataset. Each split is a list of triples <h,r,t>
    constants, predicates, ground, train, valid, test = ntp_dataset_triple(dataset, "../data")
    num_variables = 2 * len(predicates)

    np.random.seed(0)
    tf.random.set_seed(0)


    def get_all_fragments_dict(facts):

        fragment_dict = OrderedDict()
        for h,t in combinations(range(len(constants)), 2):
            if key(h, t) not in fragment_dict:
                fragment_dict[key(h, t)] = Fragment(h, t)
        for h, r, t in facts:
            fragment_dict[key(h, t)].add_rel(h, t, r)

        return fragment_dict



    # fragment_dict = get_connected_fragment_dict(train)
    fragment_dict = get_all_fragments_dict(train)
    fragment_interpretations = [f.get_interpretation(depth=num_variables) for f in fragment_dict.values()]
    fragment_indices = list(fragment_dict.keys())
    num_connected = len(fragment_dict)


    y = np.array(fragment_interpretations).astype(np.bool)
    indices = np.array(fragment_indices)

    num_examples =  num_connected

    sample_indices = np.tile(np.expand_dims(indices,axis=1), [1, num_samples, 1])


    P = KBCTractablePotential(hidden_layers, len(constants), embedding_size=embedding_size, num_variables=num_variables)
    P.beta = tf.ones(())
    sampler = nmln.inference.GPUGibbsSamplerBool(potential=P,
                                            num_examples=num_examples,
                                            num_variables=num_variables,
                                            num_chains=num_samples)
    adam = tf.keras.optimizers.Adam(lr)



    def train_step(i):

        for _ in range(inter_sample_burn):
            neg_data = sampler.sample(sample_indices)

        noise = tf.random.uniform(shape=y.shape)
        pos_data = tf.where(noise > p_noise, y, tf.logical_not(y))


        with tf.GradientTape() as tape:
            positive_potentials = P(tf.cast(pos_data, tf.float32), indices)
            positive_potentials = tf.reduce_mean(positive_potentials)
            negative_potentials = P(tf.cast(neg_data, tf.float32), sample_indices)
            negative_potentials = tf.reduce_mean(negative_potentials)
            loss = negative_potentials - positive_potentials

        gradients = tape.gradient(target=loss,sources=P.variables)
        grads_vars = zip(gradients, P.variables)
        adam.apply_gradients(grads_vars)

        return neg_data.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()


    #Evaluation Stuff
    #Fragment_indices_to_row_in_marginals
    d = {}
    for (i, j) in fragment_indices:
        d[(i, j)] = len(d)

    MARG = np.zeros(y.shape)
    valid_rank = Ranking(test_triples=valid,
                         all_triples=ground,
                         entity_indices=np.array(range(len(constants))))

    test_rank = Ranking(test_triples=test,
                         all_triples=ground,
                         entity_indices=np.array(range(len(constants))))


    for i in range(0, 5000):

        samples, positive_potentials, negative_potentials, loss = train_step(i)
        MARG += np.sum(samples, axis=1)
        if i%10==0:
            M = MARG / ((i + 1) * num_samples)
            scoring_function = NTPLikeScoringFunction(M, d)
            valid_res, updated = valid_rank.evaluation(scoring_function)
            if updated:
                test_res, _ = test_rank.evaluation(scoring_function)
            print(valid_res, test_res)
    return valid_res, test_res



if __name__=="__main__":

    RES = []

    # NATIONS
    # lr=1e-3
    # NUM_SAMPLES = (10,)
    # EMBEDDING_SIZES = (10,)
    # P_NOISES = (0.03,)
    # HIDDEN_LAYERS = [((75, tf.nn.relu),(50, tf.nn.relu))]

    TRIALS = [0]
    DATASETS = ["kinship", "umls"]
    NUM_SAMPLES = (10,)
    EMBEDDING_SIZES = (10,100)
    P_NOISES = (0.02,0.03, 0.05, 0.1)
    LRS = [1e-2, 1e-3]
    HIDDEN_LAYERS = [((75, tf.nn.relu),(50, tf.nn.relu)), ((50, tf.nn.relu),(50, tf.nn.relu),(50, tf.nn.relu))]


    for A in product(TRIALS, DATASETS, NUM_SAMPLES, EMBEDDING_SIZES, P_NOISES, LRS, HIDDEN_LAYERS):
        _, dataset, num_samples,embedding_size,p_noise,lr, hidden_layers = A
        r = main(dataset,num_samples,embedding_size,p_noise,lr,hidden_layers)
        RES.append((A, r))
        for i in RES: print(i)
        with open("res_nmlns_ntp_new", "a") as f:
            for i in RES:
                f.write(str(i))
                f.write("\n")






