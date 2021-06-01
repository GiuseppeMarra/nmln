import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import datasets
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
import mme


inter_sample_burn = 1
n_sample = 14

class NTPLikeScoringFunction():

    def __init__(self, marginal, ontology):
        self.marginal = marginal
        self.ontology = ontology
        self.domain = self.ontology._domain_list[0]


    def __call__(self, corrupted):
        scores = []
        for h,r,t in corrupted:
            size = self.domain.num_constants
            h = self.domain.constant_name_to_id[h]
            t = self.domain.constant_name_to_id[t]
            scores.append(self.marginal[self.ontology._predicate_range[r][0] + h * size + t])
        return scores


def create_mappings(ontology_small, ontology_all):
    domain_small = ontology_small._domain_list[0]
    domain_all = ontology_all._domain_list[0]
    mapping = []
    for i in range(ontology_small.linear_size()):
        p, [c1,c2] = ontology_small.id_to_atom(i)
        c1 = domain_small.domains[c1]
        c2 = domain_small.domains[c2]
        c1_all = domain_all.constant_name_to_id[c1]
        c2_all = domain_all.constant_name_to_id[c2]
        mapping.append(ontology_all.atom_to_id(p, [c1_all,c2_all]))
    return mapping


def create_hb_interpretation_from_triples(ontology, triples):

    hb = np.zeros([1, ontology.linear_size()])
    d_small = ontology._domain_list[0]
    n_small = d_small.num_constants
    set_constants_small = set(d_small.domains)
    for h, r, t in triples:
        if h in set_constants_small and t in set_constants_small:
            h_small = d_small.constant_name_to_id[h]
            t_small = d_small.constant_name_to_id[t]
            hb[0, ontology._predicate_range[r][0] + h_small * n_small + t_small] = 1
    return hb.astype(np.bool)


def filter_triples(triples, domain):
    res = []
    constants = set(domain.domains)
    for h, r, t in triples:
        if h in constants and t in constants:
            res.append((h,r,t))
    return res



def main(dataset, num_samples, embedding_size, p_noise, lr, hidden_layers,flips, log_file):

    # Loading the dataset. Each split is a list of triples <h,r,t>
    constants, predicates, ground, train, valid, test = datasets.ntp_dataset_triple(dataset)

    np.random.seed(0)
    tf.random.set_seed(0)


    constants = [k for k in constants.values()]
    d = mme.Domain("nations", num_constants=len(constants), constants=constants)

    predicates_list = []
    for p,k in predicates.items():
        predicates_list.append(mme.Predicate(name=k, domains=(d,d)))

    o = mme.Ontology(domains=[d], predicates=predicates_list)
    MARG = np.zeros([o.linear_size()], dtype=np.float64)
    count_all = np.zeros([o.linear_size()], dtype=np.float64)
    testing = True
    if testing:
        valid_rank = Ranking(test_triples=valid,
                             all_triples=ground,
                             entity_indices=np.array(constants))


        test_rank = Ranking(test_triples=test,
                            all_triples=ground,
                            entity_indices=np.array(constants))
        test_res = 0.


    # This for-loop iterates over random samples of the constants.
    # Each of the iterations is identical to 1 run in nations
    # Scores are averaged
    iteration = 0
    for _ in range(-1,100000):

        constants_small = np.random.choice(constants, size=n_sample, replace=False)
        d_small = mme.Domain("nations", num_constants=len(constants_small), constants=constants_small)
        num_variables = (len(constants_small) ** 2) * len(predicates)

        ground_small = filter_triples(ground, d_small)
        valid_small = filter_triples(valid, d_small)
        test_small = filter_triples(test, d_small)
        if len(valid_small) == 0 or len(test_small) == 0:
            print()
            continue

        iteration+=1


        predicates_list = []
        for p, k in predicates.items():
            predicates_list.append(mme.Predicate(name=k, domains=(d_small, d_small)))

        o_small = mme.Ontology(domains=[d_small], predicates=predicates_list)




        mapping = create_mappings(ontology_small=o_small, ontology_all=o)
        y = create_hb_interpretation_from_triples(o_small, train)
        y_ground = create_hb_interpretation_from_triples(o_small, ground)
        y = tf.constant(y, dtype=tf.bool)



        idx_of, ids_of = o_small.one_factors()


        # Removing reflexive atoms from one-factors
        n = len(constants_small)
        a = np.reshape(idx_of, [n-1, n//2, len(predicates), 4])
        a = a[:,:,:,1:3]
        idx_of = np.transpose(np.reshape(a, [a.shape[0], a.shape[1], -1]), [1,0,2])
        ids_of = np.transpose(ids_of, [1,0,2])





        if embedding_size > 0:
            P = KBCPotential(3, o_small, hidden_layers, len(constants_small), embedding_size=embedding_size)
        else:
            P = KBCPotentialNoConstantEmb(3, o_small, hidden_layers, len(constants_small))
        P.beta = 1.
        sampler = mme.inference.FactorizedGPUGibbsSamplerBool(potential=P,
                                                              factorization_ids=ids_of,
                                                              factorization_idx=idx_of,
                                                              num_examples=1,
                                                              num_variables=num_variables,
                                                              num_chains=num_samples,
                                                              flips = flips)
        adam = tf.keras.optimizers.Adam(lr)


        if embedding_size <= 0:
            sampler_test = mme.inference.FactorizedGPUGibbsSamplerBool(potential=P,
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
                # positive_potentials = tf.reduce_mean(positive_potentials)
                negative_potentials = P(tf.cast(neg_data, tf.float32))
                negative_potentials = tf.reduce_mean(negative_potentials,axis= 1)
                loss = tf.reduce_mean(negative_potentials - positive_potentials)

                # positive_potentials = P.call_on_groundings(P.ground(tf.cast(pos_data, tf.float32)))
                # negative_potentials = P.call_on_groundings(P.ground(tf.cast(molecules, tf.float32)))
                # negative_potentials = tf.reduce_mean(negative_potentials,axis=1)
                # loss = negative_potentials - positive_potentials
                # loss = tf.reduce_sum(loss)

            gradients = tape.gradient(target=loss,sources=P.variables)
            grads_vars = zip(gradients, P.variables)
            adam.apply_gradients(grads_vars)

            return positive_potentials, negative_potentials, loss

        def train_step(i):

            for _ in range(inter_sample_burn):
                neg_data = sampler.sample()

            # hamming = tf.reduce_mean(tf.reduce_sum(tf.cast(molecules != y_ground, tf.float32), axis=-1))
            hamming = 0.

            positive_potentials, negative_potentials, loss = parameters_update(y,neg_data)



            if embedding_size<=0:
                neg_data = sampler_test.sample()

            # print("Train step time", time.time() - start)
            # return molecules.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy(), hamming.numpy()
            return neg_data.numpy(), 0., 0., loss.numpy(), hamming

        # Evaluation Staff
        # Fragment_indices_to_row_in_marginals

        if testing:
            valid_rank_small = Ranking(test_triples=valid_small,
                                 all_triples=ground_small,
                                 entity_indices=constants_small)

            test_rank_small = Ranking(test_triples=test_small,
                                all_triples=ground_small,
                                entity_indices=constants_small)
            test_res_small = 0.

        MARG_small = np.zeros(o_small.linear_size())
        internal_iters = 10000
        for i in range(0, internal_iters):
            start = time.time()
            samples, positive_potentials, negative_potentials, loss, hamming = train_step(i)
            duration = time.time() - start
            MARG_small += np.squeeze(np.sum(samples, axis=1))
            # print(i, "Pos:%groundings\t\t Neg:%groundings\t\t Loss:%groundings,\t\t Hamming:%groundings" % ( positive_potentials, negative_potentials, -loss, hamming))
            # print(i, positive_potentials, negative_potentials, loss)
            # print(i, "Pos:%groundings\t\t Neg:%groundings\t\t Loss:%groundings,\t\t Hamming:%groundings" % ( positive_potentials, negative_potentials, -loss, hamming))
            if i % 100 == 0 and testing:
                M = MARG_small / ((i + 1) * num_samples)
                scoring_function = NTPLikeScoringFunction(M, ontology=o_small)
                valid_res_small, updated = valid_rank_small.evaluation(scoring_function)
                if updated:
                    test_res_small, _ = test_rank_small.evaluation(scoring_function)
                    print(i, valid_res_small, test_res_small)
                print(i, positive_potentials, negative_potentials, loss, valid_res_small, test_res_small)


        MARG[mapping] += MARG_small

        count_all[mapping] += np.ones_like(MARG_small) * (internal_iters*num_samples)
        M = MARG / np.where(count_all>0, count_all, np.ones_like(count_all))
        scoring_function = NTPLikeScoringFunction(M, ontology=o)
        valid_res, updated = valid_rank.evaluation(scoring_function)
        test_res, _ = test_rank.evaluation(scoring_function)
        print()
        print()
        print()
        print(np.sum(M == 0))
        print(iteration, valid_res, test_res)
        print()
        print()
        print()
        print()
        with open(log_file, "a") as f:
            f.write(str(iteration) + str(valid_res) + str(test_res))
            f.write("\n")





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
    DATASETS = ["umls"]
    NUM_SAMPLES = (10,)
    EMBEDDING_SIZES = (0,)
    P_NOISES = (0.01,)
    LRS = [1e-5]
    HIDDEN_LAYERS = [[(100, tf.nn.relu)]]
    FLIPS = [10]










    for A in product(TRIALS, FLIPS,DATASETS, NUM_SAMPLES, EMBEDDING_SIZES, P_NOISES, LRS, HIDDEN_LAYERS):
        _, flips, dataset, num_samples,embedding_size,p_noise,lr, hidden_layers = A
        embbb = "-e" if embedding_size > 0 else ""
        log_file = "%s-sampling-%s_%s" % (dataset, embbb, str(time.time()))

        with open(log_file, "a") as f:
            f.write(str(A))
            f.write("\n")
        r = main(dataset,num_samples,embedding_size,p_noise,lr,hidden_layers,flips, log_file)
        RES.append((A, r))
        for i in RES: print(i)
        # with open("res_nmln%s_ntp_%s" % (embbb, dataset), "a") as groundings:
        #     for i in RES:
        #         groundings.write(str(i))
        #         groundings.write("\n")






