import numpy as np
from itertools import combinations, permutations
from collections import OrderedDict, defaultdict
import random
import nmln
import tensorflow as tf
from typing import List, Tuple, Dict, Callable, Optional


class Ranking():

    def __init__(self,
                 test_triples: List[Tuple[str, str, str]],
                 all_triples: List[Tuple[str, str, str]],
                 entity_indices: np.ndarray):

        self.test_triples = OrderedDict()  # used as a set
        for t in (test_triples):
            self.test_triples[t] = True
        self.all_triples = OrderedDict()  # used as a set
        for t in (all_triples):
            self.all_triples[t] = True

        entities = entity_indices.tolist()

        self.corruptions = {}
        for s_idx, p_idx, o_idx in test_triples:
            corrupted_subject = [(entity, p_idx, o_idx) for entity in entities if
                                 (entity, p_idx, o_idx) not in self.all_triples or entity == s_idx]
            corrupted_object = [(s_idx, p_idx, entity) for entity in entities if
                                (s_idx, p_idx, entity) not in self.all_triples or entity == o_idx]

            index_l = corrupted_subject.index((s_idx, p_idx, o_idx))
            index_r = corrupted_object.index((s_idx, p_idx, o_idx))

            nb_corrupted_l = len(corrupted_subject)
            # nb_corrupted_r = len(corrupted_object)

            corrupted = corrupted_subject + corrupted_object

            self.corruptions[s_idx, p_idx, o_idx] = (index_l, index_r, corrupted, nb_corrupted_l)

        self.max = {"hits@1": 0.0}

    def evaluation(self, scoring_function: Callable):

        hits = dict()
        hits_at = [1, 3, 5, 10]

        for hits_at_value in hits_at:
            hits[hits_at_value] = 0.0

        def hits_at_n(n_, rank):
            if rank <= n_:
                hits[n_] = hits.get(n_, 0) + 1

        counter = 0
        mrr = 0.0

        for s_idx, p_idx, o_idx in self.test_triples:

            (index_l, index_r, corrupted, nb_corrupted_l) = self.corruptions[s_idx, p_idx, o_idx]

            scores_lst = scoring_function(corrupted)

            scores_l = scores_lst[:nb_corrupted_l]
            scores_r = scores_lst[nb_corrupted_l:]

            rank_l = 1 + np.argsort(np.argsort(- np.array(scores_l)))[index_l]
            counter += 1

            for n in hits_at:
                hits_at_n(n, rank_l)

            mrr += 1.0 / rank_l

            rank_r = 1 + np.argsort(np.argsort(- np.array(scores_r)))[index_r]
            counter += 1

            for n in hits_at:
                hits_at_n(n, rank_r)

            mrr += 1.0 / rank_r

        counter = float(counter)

        mrr /= counter

        for n in hits_at:
            hits[n] /= counter

        metrics = dict()
        metrics['MRR'] = mrr
        for n in hits_at:
            metrics['hits@{}'.format(n)] = hits[n]

        updated = False
        if hits[1] >= self.max["hits@1"]:
            self.max = metrics
            updated = True

        return (mrr, hits[1], hits[3], hits[5], hits[10]), updated


class NTPLikeScoringFunction():

    def __init__(self, marginal, ontology):
        self.marginal = marginal
        self.ontology = ontology

    def __call__(self, corrupted):
        scores = []
        for h, r, t in corrupted:
            size = self.ontology.predicates[r].domains[0].num_constants
            scores.append(self.marginal[0, self.ontology._predicate_range[r][0] + h * size + t])
        return scores


def shuffled(sequence):
    deck = sequence
    count = len(deck)
    while count>0:
        count-=1
        i = random.randint(0, len(deck) - 1)
        yield deck[i]


def key(i, j):
    return min(i, j), max(i, j)


class Fragment():

    def __init__(self, i, j):

        self.i = min(i, j)
        self.j = max(i, j)
        self.rels_ij = []
        self.rels_ji = []

    def add_rel(self, h, t, r):
        if h < t:
            self.rels_ij.append(int(r))
        else:
            self.rels_ji.append(int(r))

    def exists_rel(self, h, t, r):
        if h < t:
            return r in self.rels_ij
        else:
            return r in self.rels_ji

    def get_interpretation(self, depth):
        i = np.zeros(depth).astype(np.int)
        i[np.concatenate((2 * np.array(self.rels_ij), 2 * np.array(self.rels_ji) + 1), axis=0).astype(np.int)] = 1
        return i

    def key(self):
        return (self.i, self.j)


def get_connected_fragment_dict(facts):
    connected_fragment_dict = OrderedDict()
    for h, r, t in facts:
        try:
            connected_fragment_dict[key(h, t)].add_rel(h, t, r)
        except KeyError:
            f = Fragment(h, t)
            f.add_rel(h, t, r)
            connected_fragment_dict[key(h, t)] = f

    return connected_fragment_dict


def get_all_fragments_dict(facts, constants):
    fragment_dict = OrderedDict()
    for h, t in combinations(range(len(constants)), 2):
        fragment_dict[key(h, t)] = Fragment(h, t)
    for h, r, t in facts:
        fragment_dict[key(h, t)].add_rel(h, t, r)

    return fragment_dict


def get_disconnected_indices_coherent_corruption(connected_fragment_dict, num_disconnected, constants):
    """ Creates num_disconnected fragments by corrupting heads and tails with other domains that
     appears in that position for the same predicate. In this way the corruption is more realistic.
     For example, the fragment (brussels, belgium) containing the atom locatedIn(brussels, belgium) can be corrupted
     into locatedIn(brussels, italy) but never with locatedIn(brussels, rome). In this way, the negative
     examples are more difficult cases for the model and can help in creating better margins between
    true and false facts."""

    #TODO Ugly implementation. Need to do it fast.

    disconnected_fragment_indices = []

    # Here, we create some indices for getting interesting fragments faster.
    rels_to_heads = defaultdict(lambda: [])
    rels_to_tails = defaultdict(lambda: [])
    for (i,j) in connected_fragment_dict:
        for r in connected_fragment_dict[(i,j)].rels_ij:
            rels_to_heads[r].append(i)
            rels_to_tails[r].append(j)
        for r in connected_fragment_dict[(i,j)].rels_ji:
            rels_to_heads[r].append(j)
            rels_to_tails[r].append(i)



    def tail_corruption(frag):

            if len(frag.rels_ij) > 0:
                for r in np.random.permutation(frag.rels_ij):
                    for n in shuffled(rels_to_tails[r]):
                        if (n != i) and key(i, n) not in connected_fragment_dict:
                            yield key(i, n)
            if len(frag.rels_ji) > 0:
                for r in np.random.permutation(frag.rels_ji):
                    for n in shuffled(rels_to_tails[r]):
                        if (n != j) and key(j, n) not in connected_fragment_dict:
                            yield key(j, n)


    def head_corruption(frag):
        if len(frag.rels_ij) > 0:
            for r in np.random.permutation(frag.rels_ij):
                for n in shuffled(rels_to_heads[r]):
                    if (n != j) and key(n, j) not in connected_fragment_dict:
                        yield key(j, n)
        if len(frag.rels_ji) > 0:
            for r in np.random.permutation(frag.rels_ji):
                for n in shuffled(rels_to_heads[r]):
                    if (n != i) and key(n, i) not in connected_fragment_dict:
                        yield key(i, n)

    for t, ((i, j), frag) in enumerate(connected_fragment_dict.items()):

        tail_gen = tail_corruption(frag)
        head_gen = head_corruption(frag)
        while len(disconnected_fragment_indices) < (t+1)*num_disconnected:
            try:
                k = tail_gen.__next__()
                disconnected_fragment_indices.append(k)
                flag1=False
            except:
                flag1=True
            try:
                k = head_gen.__next__()
                disconnected_fragment_indices.append(k)
                flag2=False
            except:
                flag2 = True
            if flag1 and flag2:
                break
    return np.reshape(disconnected_fragment_indices, [-1, 2])


def get_disconnected_indices_random_corruption(connected_fragment_dict, num_disconnected, constants):
    """ Creates num_disconnected fragments by randomly (and evenly) corrupting heads and tails of connected
     fragments with disconnected fragments. Return the fragments as pairs of indices since interpretation is
    all zeros. """

    disconnected_fragment_indices = []
    for t, k in enumerate(connected_fragment_dict):
        (i, j) = k
        while len(disconnected_fragment_indices) < (t+1)*num_disconnected // 2:
            n = random.randint(0, len(constants) - 1)
            if (n != i) and (i, n) not in connected_fragment_dict:
                disconnected_fragment_indices.append((i, n))

        # n,j negative
        while len(disconnected_fragment_indices) < (t+1)*num_disconnected:
            # n = int(np.random.choice(np.arange(len(domains)), size=1))
            n = random.randint(0, len(constants) - 1)
            if (n != j) and (n, j) not in connected_fragment_dict:
                disconnected_fragment_indices.append((n, j))

    return np.reshape(disconnected_fragment_indices, [-1, 2])


class KBCTractablePotential(nmln.potentials.Potential):

    def __init__(self, hidden_layers, num_constants, embedding_size, num_variables):
        super(KBCTractablePotential, self).__init__()
        self.embedding_size = embedding_size
        if self.embedding_size > 0:
            self.embeddings = tf.Variable(initial_value=tf.random.normal([num_constants, embedding_size]))
        self.model = tf.keras.Sequential()
        for units, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(units, activation=activation))
        self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))
        self.permutation = [a + 1 if a % 2 == 0 else a - 1 for a in range(num_variables)]

    def __call__(self, y, x=None):
        if x is None:
            raise Exception("KBCTractablePotential needs x when called")

        y_permuted = tf.gather(params=y, indices=self.permutation, axis=-1)

        if self.embedding_size > 0:
            e = tf.nn.embedding_lookup(self.embeddings, x)
            e_permuted = tf.reshape(tf.gather(params=e, indices=[1, 0], axis=-2), list(x.shape[:-1]) + [-1])
            e = tf.reshape(e, list(x.shape[:-1]) + [-1])
            input_permuted = tf.concat((y_permuted, e_permuted), axis=-1)
            input = tf.concat((y, e), axis=-1)
            input = tf.stack((input, input_permuted), axis=0)
        else:
            input = tf.stack((y, y_permuted), axis=0)

        a = tf.squeeze(self.model(input))
        return tf.reduce_sum(a, axis=0)
        # return a


class KBCPotential(nmln.potentials.CountableGroundingPotential):

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
            self.groundings_hb_indices, self. indices = ontology.all_fragments_idx(k, get_ids=True)
            self.groundings_hb_indices = tf.constant(self.groundings_hb_indices)
            self.indices = tf.constant(self.indices)
        @tf.function
        def ground(self, y, x=None):
            y = tf.gather(params=y, indices=self.groundings_hb_indices, axis=-1)
            e = tf.nn.embedding_lookup(self.embeddings_entities, self.indices)
            e = tf.reshape(e, [1] + list(self.indices.shape[:-1]) + [-1])
            if len(y.shape) > 3:  # if there is also the sampling dimension (i.e. when called by GibbsSampling)
                e = tf.tile(tf.expand_dims(e, axis=1), [1, y.shape[1], 1, 1])
            y = self.embeddings_relations(y)
            input = tf.concat((y, e), axis=-1)
            return input

        @tf.function
        def call_on_groundings(self, y, x=None):
            a = tf.squeeze(self.model(y), axis=-1)  # remove potential value 1 dimension
            return a

        @tf.function
        def reduce_groundings(self, y):
            return tf.reduce_sum(y, axis=-1)



class KBCPotentialNoConstantEmb(nmln.potentials.CountableGroundingPotential):

        def __init__(self, k, ontology, hidden_layers, num_constants):
            super(KBCPotentialNoConstantEmb, self).__init__()
            self.model = tf.keras.Sequential()
            for units, activation in hidden_layers:
                self.model.add(tf.keras.layers.Dense(units, activation=activation))
            self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))
            self.groundings_hb_indices = tf.constant(ontology.all_fragments_idx(k))


        @tf.function
        def ground(self, y, x=None, subsampling = None):
            # if subsampling is not None:
            #     id_sample = tf.random.uniform(shape = [int(len(ontology.groundings_hb_indices) * subsampling)], minval=0,
            #                                               maxval=len(ontology.groundings_hb_indices), dtype=tf.int32)
            #     groundings_hb_indices = tf.gather(ontology.groundings_hb_indices, id_sample, axis=0)
            # else:
            #     groundings_hb_indices = ontology.groundings_hb_indices
            groundings_hb_indices = self.groundings_hb_indices
            y = tf.gather(params=y, indices=groundings_hb_indices, axis=-1)
            return y

        @tf.function
        def call_on_groundings(self, y, x=None):
            a = tf.squeeze(self.model(y), axis=-1)  # remove potential value 1 dimension
            return a

        @tf.function
        def reduce_groundings(self, y):
            return tf.reduce_sum(y, axis=-1)

        @tf.function
        def __call__(self, y, x=None, subsampling = None):
            g = self.ground(y, x, subsampling)
            g = self.call_on_groundings(g, x)
            r = self.reduce_groundings(g)

            return r




class KBCTractablePotential2(nmln.potentials.Potential):

    def __init__(self, hidden_layers, num_constants, embedding_size, num_variables, constants_embeddings=None,
                 dropout_rate=0.):
        super(KBCTractablePotential2, self).__init__()
        self.embedding_size = embedding_size
        if self.embedding_size > 0:
            if constants_embeddings is None:
                # ontology.embeddings_entities = tf.Variable(initial_value=tf.random.normal([num_constants, embedding_size]), name="constants_embeddings")
                self.embeddings_entities = tf.Variable(tf.initializers.GlorotUniform()([num_constants, embedding_size]),
                                                       name="constants_embeddings")
            else:
                self.embeddings_entities = tf.Variable(initial_value=constants_embeddings, name="constants_embeddings")
            self.embeddings_relations = tf.keras.layers.Dense(units=self.embedding_size, activation=None,
                                                              use_bias=False)
        self.model = tf.keras.Sequential()
        for units, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(units))
            self.model.add(activation)
            if dropout_rate > 0.: self.model.add(tf.keras.layers.Dropout(dropout_rate))
        self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))
        self.permutation = [a + 1 if a % 2 == 0 else a - 1 for a in range(num_variables)]

    def __call__(self, y, x=None, training=None):
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

        a = tf.squeeze(self.model(input, training=training))
        res = tf.reduce_sum(a, axis=0)
        return res
        # return a


class KBCTractablePotentialWE(nmln.potentials.Potential):

    def __init__(self, hidden_layers, num_constants, embedding_size, num_variables, bow, we):
        super(KBCTractablePotentialWE, self).__init__()
        self.embedding_size = embedding_size
        self.bow = tf.constant(bow, dtype=tf.int32)
        self.we = tf.Variable(initial_value=we)
        self.embeddings_words = tf.keras.layers.Dense(units=self.embedding_size, activation=tf.nn.relu)
        self.embeddings_relations = tf.keras.layers.Dense(units=self.embedding_size, activation=None, use_bias=False)
        self.model = tf.keras.Sequential()
        for units, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(units, activation=activation))
        self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))
        self.permutation = [a + 1 if a % 2 == 0 else a - 1 for a in range(num_variables)]

    def __call__(self, y, x=None, training=False):
        if x is None:
            raise Exception("KBCTractablePotential needs x when called")

        y_permuted = tf.gather(params=y, indices=self.permutation, axis=-1)
        we = self.embeddings_words(self.we)
        bow = tf.gather(params=self.bow, indices=x)
        e = tf.gather(indices=bow, params=we)
        e = tf.reduce_mean(e, axis=-2)


        e_permuted = tf.reshape(tf.gather(params=e, indices=[1, 0], axis=-2), list(x.shape[:-1]) + [-1])
        e = tf.reshape(e, list(x.shape[:-1]) + [-1])
        e = tf.stack((e, e_permuted), axis=0)
        y = tf.stack((y, y_permuted), axis=0)
        y = self.embeddings_relations(y)
        input = tf.concat((y, e), axis=-1)

        a = tf.squeeze(self.model(input, training=training))
        return tf.reduce_sum(a, axis=0)


def evaluate(key_to_fragment_index_, eval_facts, marginals):
    true = np.array([(key_to_fragment_index_[key(i, j)], 2 * r if i < j else 2 * r + 1) for i, r, j in
                     eval_facts[:len(eval_facts) // 2]])
    false = np.array([(key_to_fragment_index_[key(i, j)], 2 * r if i < j else 2 * r + 1) for i, r, j in
                      eval_facts[len(eval_facts) // 2:]])

    t = marginals[true[:, 0], true[:, 1]]
    f = marginals[false[:, 0], false[:, 1]]
    return np.mean(t > f)


def find_threshold(t, f):
    from bisect import bisect_left
    t = sorted(t)
    f = sorted(f)

    best_c = 0
    best_th = 0
    for i, T in enumerate(t):
        j = bisect_left(f,T)
        c = len(t) - i + j # len(t) - i are the # of elements > T in t. j are the # of elements <=T in groundings.
        if c > best_c:
            best_c = c
            best_th = T
    return best_th

def find_threshold_old(t, f):
    # Brute force. It can be done much faster with a smarter strategy.

    best_th = 0
    best_c = 0
    for T in t:
        c = 0
        for T1 in t:
            if T1 >= T:
                c += 1
        for F1 in f:
            if F1 < T:
                c += 1
        if c > best_c:
            best_c = c
            best_th = T
    return best_th


def new_evaluate(key_to_fragment_index_, eval_facts, marginals, threshold=None):
    true = np.array([(key_to_fragment_index_[key(i, j)], 2 * r if i < j else 2 * r + 1) for i, r, j in
                     eval_facts[:len(eval_facts) // 2]])
    false = np.array([(key_to_fragment_index_[key(i, j)], 2 * r if i < j else 2 * r + 1) for i, r, j in
                      eval_facts[len(eval_facts) // 2:]])

    t = marginals[true[:, 0], true[:, 1]]
    f = marginals[false[:, 0], false[:, 1]]

    if threshold is None:
        threshold = find_threshold(t, f)

    return (np.sum(t > threshold) + np.sum(f <= threshold)) / (2 * len(t)), threshold


def new_new_evaluate(key_to_fragment_index_, eval_facts, marginals, threshold=None):
    """New evaluation routine with a per-relation threshold as in TransA paper."""

    def __inner__(l):
        d = defaultdict(lambda: [])
        for i,r,j in l:
            d[r].append((key_to_fragment_index_[key(i, j)], 2 * r if i < j else 2 * r + 1))

        for r in d.keys():
            d[r] = np.array(d[r])
        return d

    true = __inner__(eval_facts[:len(eval_facts) // 2])
    false = __inner__(eval_facts[len(eval_facts) // 2:])

    def __true_vs_all__(t,f,th=None):
        t = marginals[t[:, 0], t[:, 1]]
        f = marginals[f[:, 0], f[:, 1]]

        if th is None:
            th = find_threshold(t, f)

        return np.sum(t >= th) + np.sum(f < th), th

    th = threshold if threshold is not None else defaultdict(lambda: None)
    tot = 0
    correct = 0
    for r in true.keys():
        tr = true[r]
        fr = false[r]
        c, t =__true_vs_all__(tr, fr, th=th[r])
        tot += len(true[r]) + len(false[r])
        correct += c
        th[r] = t



    return correct/tot, th