import os
SEED = 0
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



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
import numpy as np
import random
import datasets
from collections import OrderedDict
from itertools import combinations,product
import mme
from kbc_utils import key, Fragment, get_connected_fragment_dict, get_disconnected_indices_coherent_corruption, get_disconnected_indices_random_corruption, get_all_fragments_dict, KBCTractablePotential, KBCTractablePotential2, KBCTractablePotentialWE, evaluate, new_new_evaluate, new_evaluate



inter_sample_burn = 1




def main(dataset, minibatch_size, num_disconnected_negative_samples, num_samples, embedding_size, p_noise, lr, hidden_layers):


    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Loading the dataset. Each split is a list of triples <h,r,t>
    constants, predicates, ground, train, valid, test, embeddings, bow, we  = datasets.ntn_dataset_triple_we(dataset)
    num_variables = 2 * len(predicates)


    # Reasoning with sparsity
    connected_fragment_dict = get_connected_fragment_dict(train)
    connected_fragment_interpretations = [f.get_interpretation(depth=num_variables) for f in connected_fragment_dict.values()]
    connected_fragment_indices = list(connected_fragment_dict.keys())
    num_positive = len(connected_fragment_dict)


    ## Negative sampling

    # Restricted number of negatives
    if num_disconnected_negative_samples > 0:
        disconnected_fragment_indices_random = get_disconnected_indices_random_corruption(connected_fragment_dict, num_disconnected_negative_samples*5//10, constants)
        disconnected_fragment_indices_coherent = get_disconnected_indices_coherent_corruption(connected_fragment_dict, (num_disconnected_negative_samples*5)//10, constants)
        disconnected_fragment_indices = np.concatenate((disconnected_fragment_indices_random,disconnected_fragment_indices_coherent), axis=0)
        num_negative = num_disconnected_negative_samples * num_positive
        if num_negative != len(disconnected_fragment_indices):
            if num_negative > len(disconnected_fragment_indices):
                rest = len(disconnected_fragment_indices) - num_negative
                disconnected_fragment_indices = np.concatenate((disconnected_fragment_indices,disconnected_fragment_indices[:rest]), axis=0)
            else:
                disconnected_fragment_indices = disconnected_fragment_indices[:num_negative]
        y = np.concatenate((connected_fragment_interpretations, np.zeros([num_negative, num_variables])),
                           axis=0).astype(np.bool)
        indices = np.concatenate((connected_fragment_indices, disconnected_fragment_indices), axis=0)
    #All the negatives
    else:
        y = np.array(connected_fragment_interpretations).astype(np.bool)
        indices = np.array(connected_fragment_indices).astype(np.int32)
        num_negative=0

    num_examples = num_positive + num_negative
    sample_indices = np.tile(np.expand_dims(indices, axis=1), [1, num_samples, 1])


    # Building the NMLN

    potential = 'NO-EMBEDDING'
    if potential == 'NO-EMBEDDING':
        P = KBCTractablePotential2(hidden_layers, len(constants), embedding_size=embedding_size, num_variables=num_variables, constants_embeddings=None)
    elif potential == 'INIT-WE':
        P = KBCTractablePotential2(hidden_layers, len(constants), embedding_size=embedding_size, num_variables=num_variables, constants_embeddings=embeddings)
    elif potential == 'WE': # todo this is not working
        P = KBCTractablePotentialWE(hidden_layers, len(constants), embedding_size=embedding_size, num_variables=num_variables, bow=bow, we=we)
    P.beta = tf.ones(()) # it is not needed in single potential distributions, it is already in the linear output of the net
    sampler = mme.inference.GPUGibbsSamplerBool(potential=P,
                                            num_examples=num_examples,
                                            num_variables=num_variables,
                                            num_chains=num_samples)

    adam = tf.optimizers.Adam(lr)




    """Evaluation staff"""
    eval_fragment_dict = get_connected_fragment_dict(valid+test)
    eval_fragment_indices = np.array(list(eval_fragment_dict.keys()))
    eval_fragment_samples_indices = np.tile(np.expand_dims(eval_fragment_indices,axis=1), [1, num_samples, 1])
    key_to_fragment_index_ = {k:i for i,k in enumerate(eval_fragment_dict)}
    sampler_eval = mme.inference.GPUGibbsSamplerBool(potential=P,
                                                num_examples=len(eval_fragment_indices),
                                                num_variables=num_variables,
                                                num_chains=num_samples)


    def train_step(step):
        minibatch_conn = np.random.randint(0, len(connected_fragment_dict), size=minibatch_size)
        minibatch_disconn = np.random.randint(len(connected_fragment_dict), len(indices), size=minibatch_size)
        minibatch = np.concatenate((minibatch_conn, minibatch_disconn), axis=0)




        neg_idx = tf.gather(sample_indices, minibatch)
        for _ in range(inter_sample_burn):
            neg_data = sampler.sample(neg_idx, minibatch=minibatch)

        mb = tf.gather(y, minibatch)
        noise = tf.random.uniform(shape=mb.shape)
        pos_data = tf.where(noise > p_noise, mb, tf.logical_not(mb))
        pos_idx = tf.gather(indices, minibatch)


        with tf.GradientTape() as tape:
            pos_data = tf.cast(pos_data, tf.float32)
            positive_potentials = P(pos_data, pos_idx, training=True)
            negative_potentials = P(neg_data, neg_idx, training=True)

            positive_potentials = tf.reduce_mean(positive_potentials)
            negative_potentials = tf.reduce_mean(tf.reduce_mean(negative_potentials,axis=1))
            loss = negative_potentials - positive_potentials # we invert the sign since adam always performs a descent step

        gradients = tape.gradient(target=loss,sources=P.variables)
        grads_vars = zip(gradients, P.variables)
        adam.apply_gradients(grads_vars)


        return neg_data.numpy(), positive_potentials.numpy(), negative_potentials.numpy(), loss.numpy()



    def eval_step():
        return sampler_eval.sample(tf.constant(eval_fragment_samples_indices)).numpy()


    # def eval_step():
    #     eval_minibatch_size = 100
    #     for k in range(len(eval_fragment_samples_indices)//eval_minibatch_size):
    #         eval_minibatch = list(range(k*eval_minibatch_size, (k+1)* eval_minibatch_size))
    #         eval_idx = tf.gather(eval_fragment_samples_indices, eval_minibatch)
    #         sampler_eval.sample(eval_idx, minibatch=tf.constant(eval_minibatch)).numpy()
    #         print(k, len(eval_fragment_samples_indices)//eval_minibatch_size)
    #     return sampler_eval.current_state.numpy()

    max_validation = 0
    eval_iter = 10
    MARG = np.zeros([len(eval_fragment_indices), num_variables])
    MAX_PATIENCE = 100 # it is equivalent to MAX_PATIENCE x eval_iter steps
    patience = MAX_PATIENCE
    tot_samples = 0

    for i in range(0, 10000):
        _, positive_potentials, negative_potentials, loss = train_step(i)
        if i%eval_iter==0:
            samples = eval_step()
            tot_samples += len(samples[0])
            MARG += np.sum(samples, axis=1)
            M = MARG / tot_samples
            # valid_res = evaluate(valid, M)
            valid_res, threshold = new_new_evaluate(key_to_fragment_index_, valid, M)

            print("VALID", i, valid_res, positive_potentials, negative_potentials, loss)
            if (positive_potentials == negative_potentials == loss == 0.0):
                break
            if valid_res > max_validation:
                patience = MAX_PATIENCE
                max_validation = valid_res
                # test_res = evaluate(test, M)
                test_res,_ = new_new_evaluate(key_to_fragment_index_, test, M, threshold)

                print("TEST:", i, test_res)
            else:
                patience-=1
            if patience<0:
                break

        if mme.utils.heardEnter():
            break
    test_res_old = evaluate(key_to_fragment_index_, test, M)
    print("TEST:", None, test_res_old)
    return max_validation, test_res



if __name__=="__main__":

    RES = []

    TRIALS = [0]

    # DATASETS = ["fb"]
    # MINIBATCH_SIZE = [16]
    # NEGATIVE = [6, 20, 40]
    # NUM_SAMPLES = [10]
    # EMBEDDING_SIZES = [20, 50, 100]
    # P_NOISES = [0.03, 0.1, 0.2]
    # LRS = [1e-2]
    # # HIDDEN_LAYERS = [((50, tf.keras.layers.LeakyReLU(alpha=0.2)),(50, tf.keras.layers.LeakyReLU(alpha=0.2)),(50, tf.keras.layers.LeakyReLU(alpha=0.2)),)]
    # # HIDDEN_LAYERS = [((50, tf.keras.layers.ReLU()),(50, tf.keras.layers.ReLU()),(50, tf.keras.layers.ReLU())), ((50, tf.keras.layers.ReLU()),(50, tf.keras.layers.ReLU()))]
    # # HIDDEN_LAYERS = [((50, tf.keras.layers.ReLU()),(50, tf.keras.layers.ReLU()))]
    # HIDDEN_LAYERS = [((100, tf.keras.layers.ReLU()),)]


    # MINIBATCH_SIZE = [16, 128]
    # NEGATIVE = [6, 10]
    # NUM_SAMPLES = [10]
    # EMBEDDING_SIZES = [20,100]
    # P_NOISES = [0.1, 0.01, 0.05]
    # LRS = [1e-2]

    # DATASETS = ["fb"]
    # MINIBATCH_SIZE = [16, 128]
    # NEGATIVE = [6, 10]
    # NUM_SAMPLES = [10]
    # EMBEDDING_SIZES = [20,100]
    # P_NOISES = [0.1, 0.01, 0.05]
    # LRS = [1e-2]

    DATASETS = ["wn"]
    MINIBATCH_SIZE = [512]
    NEGATIVE = [10]
    NUM_SAMPLES = [10]
    EMBEDDING_SIZES = [20, 50]
    P_NOISES = [0.03,0.04, 0.06]
    LRS = [1e-2]



    # HIDDEN_LAYERS = [((50, tf.keras.layers.LeakyReLU(alpha=0.2)),(50, tf.keras.layers.LeakyReLU(alpha=0.2)),(50, tf.keras.layers.LeakyReLU(alpha=0.2)),)]
    HIDDEN_LAYERS = [[[50, tf.keras.layers.ReLU()]], [[50, tf.keras.layers.ReLU()],[50, tf.keras.layers.ReLU()]]]
    # HIDDEN_LAYERS = [[[50, tf.keras.layers.Activation(tf.keras.activations.sigmoid)]],]
    # HIDDEN_LAYERS = [((50, tf.keras.layers.ReLU()),(50, tf.keras.layers.ReLU()))]


    r = (1.,1.)
    for A in product(DATASETS, MINIBATCH_SIZE, NEGATIVE, NUM_SAMPLES, EMBEDDING_SIZES, P_NOISES, LRS, HIDDEN_LAYERS,TRIALS):
        if A[-1]>0 and r[1]<0.6: continue
        dataset, minibatch_size, num_disconnected_negative_samples, num_samples,embedding_size,p_noise,lr, hidden_layers,_ = A
        r = main(dataset,minibatch_size,num_disconnected_negative_samples,num_samples,embedding_size,p_noise,lr,hidden_layers)
        RES.append((A, r))
        for i in RES: print(i)
        with open("res_nmlns_"+dataset+"_random_disconnected_accuracy", "a") as f:
            f.write(str(RES[-1]))
            f.write("\n")







