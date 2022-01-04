import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from itertools import combinations,product,permutations
import tensorflow as tf
from kbc_utils import Ranking, KBCPotential, KBCPotentialNoConstantEmb
import time
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
    constants, predicates, ground, train, valid, test = ntp_dataset_triple(dataset, "../data")
    num_variables = (len(constants)**2) * len(predicates)

    np.random.seed(0)
    tf.random.set_seed(0)


    constants = [k for k in constants.values()]
    d = nmln.Domain("nations", num_constants=len(constants), constants=constants)

    predicates_list = []
    for p,k in predicates.items():
        predicates_list.append(nmln.Predicate(name=k, domains=(d,d)))

    o = nmln.Ontology(domains=[d], predicates=predicates_list)


    # Getting one-factors (interpretations and ids of relative k=3 fragments
    idx_of, ids_of = o.one_factors()


    # Removing reflexive atoms from one-factors
    n = len(constants)
    a = np.reshape(idx_of, [n-1, n//2, len(predicates), 4])
    a = a[:,:,:,1:3]
    idx_of = np.transpose(np.reshape(a, [a.shape[0], a.shape[1], -1]), [1,0,2])
    ids_of = np.transpose(ids_of, [1,0,2])



    def create_hb_interpretation_from_triples(ontology,triples):

        hb = np.zeros([1,ontology.herbrand_base_size])
        for h,r,t in triples:
            hb[0, ontology._predicate_range[r][0] + h * d.num_constants + t] = 1
        return hb



    y = create_hb_interpretation_from_triples(o, train).astype(np.bool)
    y = tf.constant(y, dtype=tf.bool)


    if embedding_size > 0:
        P = KBCPotential(3, o, hidden_layers, len(constants), embedding_size=embedding_size)
    else:
        P = KBCPotentialNoConstantEmb(3, o, hidden_layers, len(constants))

    sampler = nmln.inference.FactorizedGPUGibbsSamplerBool(potential=P,
                                                          factorization_ids=ids_of,
                                                          factorization_idx=idx_of,
                                                          num_examples=1,
                                                          num_variables=num_variables,
                                                          num_chains=num_samples,
                                                          flips = flips)
    adam = tf.keras.optimizers.Adam(lr)


    if embedding_size <= 0:
        sampler_test = nmln.inference.FactorizedGPUGibbsSamplerBool(potential=P,
                                                              factorization_ids=ids_of,
                                                              factorization_idx=idx_of,
                                                              num_examples=1,
                                                              num_variables=num_variables,
                                                              num_chains=num_samples,
                                                              flips=flips,
                                                              initial_state=y,
                                                              evidence=y,
                                                              evidence_mask=y)


    @tf.function
    def parameters_update(pos_data, neg_data):
        noise = tf.random.uniform(shape=pos_data.shape)
        pos_data = tf.where(noise > p_noise, pos_data, tf.logical_not(pos_data))

        with tf.GradientTape() as tape:

            positive_potentials = P(tf.cast(pos_data, tf.float32))
            positive_potentials = tf.reduce_mean(positive_potentials)
            negative_potentials = P(tf.cast(neg_data, tf.float32))
            negative_potentials = tf.reduce_mean(negative_potentials)
            loss = negative_potentials - positive_potentials

        gradients = tape.gradient(target=loss,sources=P.variables)
        grads_vars = zip(gradients, P.variables)
        adam.apply_gradients(grads_vars)

        return positive_potentials, negative_potentials, loss

    def train_step(i):

        # start = time.time()
        for _ in range(inter_sample_burn):
            neg_data = sampler.sample()



        positive_potentials, negative_potentials, loss = parameters_update(y,neg_data)



        if embedding_size<=0:
            neg_data = sampler_test.sample()

        # print("Train step time", time.time() - start)
        return neg_data.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()


    #Evaluation Staff
    #Fragment_indices_to_row_in_marginals

    MARG = np.zeros(y.shape)
    testing = True
    if testing:
        valid_rank = Ranking(test_triples=valid,
                             all_triples=ground,
                             entity_indices=np.array(range(len(constants))))

        test_rank = Ranking(test_triples=test,
                             all_triples=ground,
                             entity_indices=np.array(range(len(constants))))

    test_res = 0
    valid_res = 0
    for i in range(0, 200000):

        samples, positive_potentials, negative_potentials, loss = train_step(i)
        MARG += np.sum(samples, axis=1)
        # print(i, positive_potentials, negative_potentials, loss)
        if i%10==0 and testing:
            M = MARG / ((i + 1) * num_samples)
            scoring_function = NTPLikeScoringFunction(M, ontology= o)
            valid_res, updated = valid_rank.evaluation(scoring_function)
            if updated:
                test_res, _ = test_rank.evaluation(scoring_function)
                print(i, valid_res, test_res)
            print(i, positive_potentials, negative_potentials, loss, valid_res, test_res)
    return valid_res, test_res



if __name__=="__main__":

    RES = []

    # NATIONS
    # lr=1e-3
    # NUM_SAMPLES = (10,)
    # EMBEDDING_SIZES = (10,)
    # P_NOISES = (0.03,)
    # HIDDEN_LAYERS = [((75, tf.nn.relu),(50, tf.nn.relu))]

    # TRIALS = [0]
    # DATASETS = ["nations"]
    # NUM_SAMPLES = (10,)
    # EMBEDDING_SIZES = (0,)
    # P_NOISES = (0.01,)
    # LRS = [1e-4]
    # HIDDEN_LAYERS = [((150, tf.nn.sigmoid),(100, tf.nn.sigmoid))]
    # FLIPS = [10]


    TRIALS = [0]
    DATASETS = ["nations"]
    NUM_SAMPLES = (10,)
    EMBEDDING_SIZES = (20,)
    P_NOISES = (0.01,)
    LRS = [1e-5]
    HIDDEN_LAYERS = [[(150, tf.nn.relu), (75, tf.nn.relu)]]
    FLIPS = [10]








    for A in product(TRIALS, FLIPS,DATASETS, NUM_SAMPLES, EMBEDDING_SIZES, P_NOISES, LRS, HIDDEN_LAYERS):
        _, flips, dataset, num_samples,embedding_size,p_noise,lr, hidden_layers = A
        r = main(dataset,num_samples,embedding_size,p_noise,lr,hidden_layers,flips)
        RES.append((A, r))
        for i in RES: print(i)
        embbb = "-e" if embedding_size > 0 else ""
        with open("res_nmln%s_ntp_%s" % (embbb, dataset), "a") as f:
            for i in RES:
                f.write(str(i))
                f.write("\n")






