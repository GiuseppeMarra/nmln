import tensorflow as tf
import abc
import numpy as np
import nmln


class Potential(tf.Module):

    def __init__(self):
        tf.Module.__init__(self)
        self.beta = tf.Variable(initial_value=tf.zeros(shape=()))

    @property
    @abc.abstractmethod
    def cardinality(self):
        return 1

    def __call__(self, y, x=None):
        pass

    def _reduce_(self,y,x=None):
        return self.beta * self.__call__(y,x)


class CountableGroundingPotential(Potential):

    def __init__(self):
        super(CountableGroundingPotential, self).__init__()

    @abc.abstractmethod
    def ground(self, y, x=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def call_on_groundings(self, y, x=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def reduce_groundings(self, y):
        raise NotImplementedError()

    @tf.function
    def __call__(self, y, x=None):

        g = self.ground(y,x)
        g = self.call_on_groundings(g,x)
        r = self.reduce_groundings(g)

        return r

    @abc.abstractmethod
    def _num_groundings(self):
        raise  NotImplementedError()

    @property
    def num_groundings(self):
        return self._num_groundings()


class NeuralPotential(Potential):

    def __init__(self, model):
        super(NeuralPotential, self).__init__()
        self.model = model

    def __call__(self, y, x=None):
        if x is not None:
            y = tf.concat([y,x], axis=-1)
        return self.model(y)


class GlobalPotential():

    def __init__(self, potentials=()):
        super(GlobalPotential, self).__init__()
        self.potentials = list(potentials)

    @property
    def variables(self):
        v = []
        for p in self.potentials:
            v.extend(p.variables)
        return v

    def add(self, potential):
        self.potentials.append(potential)

    def __call__(self, y, x=None):
        res = 0.
        for Phi in self.potentials:
            n = Phi._reduce_(y,x)
            res = res + n
        return res


    def save(self, path):
        print(self.variables)
        ckpt = tf.train.Checkpoint(obj = self)
        ckpt.save(path)

    def restore(self, path):
        ckpt = tf.train.Checkpoint(obj = self)
        ckpt.restore(path)


class LogicPotential(CountableGroundingPotential):

    def __init__(self, formula, logic):
        super(LogicPotential, self).__init__()
        self.formula = formula
        self.logic = logic

    @property
    def cardinality(self):
        return len(self.formula.atoms)

    def _num_grounding(self):
        return self.formula.num_groundings

    def ground(self, y, x=None):
        return self.formula.ground(herbrand_interpretation=y) # num_examples, num_groundings, 1, num_variables_in_grounding

    def call_on_groundings(self, y, x=None, evidence=None, mask_evidence=None, reduce=True):
        t = self.formula.compile(groundings=y, logic=self.logic) # num_examples, num_groundings, num_possible_assignment_to_groundings
        t = tf.cast(t, tf.float32)
        if reduce:
            return tf.reduce_sum(t, axis=-1)
        else:
            return t


class NeuralMLPPotential(CountableGroundingPotential):

    def __init__(self, k, ontology, hidden_layers, num_sample_frags = -1):
        super(NeuralMLPPotential, self).__init__()
        self.model = tf.keras.Sequential()
        for units, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(units, activation=activation))
        self.model.add(tf.keras.layers.Dense(1, use_bias=False, activation=None))
        self.num_sample_frags = num_sample_frags
        self.groundings_hb_indices = ontology.all_fragments_idx(k=k)

    def ground(self, y, x=None):
        if self.num_sample_frags > 0:
            idx = tf.random.shuffle(self.groundings_hb_indices)
            idx = idx[:self.num_sample_frags]
        else:
            idx = self.groundings_hb_indices
        y = tf.gather(params=y, indices=idx, axis=-1)
        return y

    def call_on_groundings(self, y, x=None, evidence=None, mask_evidence=None, reduce=True):
        input = y
        a = tf.squeeze(self.model(input))  # remove potential value 1 dimension
        return a

    def reduce_groundings(self, y):
        return tf.reduce_sum(y, axis=-1)  # aggregate fragments


class HierachicalPotential(Potential):

    def __init__(self, n, k, ontology, hidden_layers):
        super(HierachicalPotential, self).__init__()
        self.idx, self.ids = ontology.all_fragments_idx(k=k, get_ids=True)
        self.layers = [tf.keras.layers.Dense(u, activation=activation) for u, activation in hidden_layers]
        output_layer = [tf.keras.layers.Dense(1, activation=None, use_bias=False)]
        self.layers = self.layers + output_layer


        self.ids_for_gather = self.ids + n * np.arange(k)

        ids = np.reshape(self.ids_for_gather, [1] + list(self.ids_for_gather.shape))
        c = np.equal(np.reshape(np.arange(n * k), [n * k, 1, 1]), ids)
        c = np.any(c, axis=-1)
        self.aggregation_scheme  = np.reshape(np.where(c)[1], [n * k, -1])


    def ground(self, y, x=None):
        idx = self.groundings_hb_indices
        y = tf.gather(params=y, indices=idx, axis=-1)
        return y

    def call_on_groundings(self, y, x=None, evidence=None, mask_evidence=None, reduce=True):
        for i, hidden in enumerate(self.layers):
            if i > 0:
                h = tf.gather(params=last, indices=self.ids_for_gather, axis=-2)
                h = tf.reshape(h, list(y.shape[:-1]) + [-1])
                x = tf.concat((y, h), axis=-1)
            else:
                x = y
            h = hidden(x)
            last = tf.gather(params=h, indices=self.aggregation_scheme, axis=-2)
            last = tf.reduce_sum(last, axis=-2)
        return tf.squeeze(h)

    def reduce_groundings(self, y):
        return tf.reduce_sum(y, axis=-1)  # aggregate fragments


class OneOfNConstraint(Potential):

    def __init__(self, list_predicates, ontology, symmetric_and_antireflexive=False, alsoNone = False):


        self.symmetric_and_antireflexive = symmetric_and_antireflexive
        self.alsoNone = alsoNone
        self.alternatives = len(list_predicates)
        self.ontology = ontology

        super(OneOfNConstraint, self).__init__()
        # Check shapes
        s = ontology._dict_indices[list_predicates[0]].shape
        try:
            N = ontology.num_constants
        except:
            raise Exception("OneOfN constraints not allowed in multi-sorted logics.")
        for l in list_predicates:
            if not np.all(ontology._dict_indices[l].shape == s):
                raise Exception("Mutual exclusivity is not allowed among these predicates. Non compatible arity.")

        arity = len(s)

        # Working on a generic predicate of the group
        if arity == 2 and self.symmetric_and_antireflexive:
            to_filter = np.reshape(np.arange(N ** arity), [N for _ in range(arity)])

            # Upper traingular indices (removing diagonal = no reflexive)
            res = []
            mask1 = np.reshape(np.triu(to_filter, 1), [-1])

            # Simmetric indices
            mask2 = np.reshape(np.triu(to_filter.T, 1), [-1])

            # Mapping indices with their symmetrical
            for i in range(len(mask1)):
                if mask1[i] != 0:
                    res.append([mask1[i], mask2[i]])
        else:
            res = np.reshape(np.arange(N ** arity), [-1, 1])


        # Repeating the current indexing scheme for all the predicates (for mutual exclusivity among them)
        res = np.tile(np.expand_dims(res, -2), [1, len(list_predicates), 1])
        base = [[[np.reshape(ontology._dict_indices[l], [-1])[0]] for l in list_predicates]]
        binary_indices = res + base
        self.indices = binary_indices
        # Translating indices in the boolean scheme (one_hot)
        binary_indices = np.eye(ontology.linear_size())[binary_indices]

        # Combining boolean values for symmetrical states (in all but arity=2 is a size=1 dimension)
        self.binary_masks = np.sum(binary_indices, axis=-2)

    def __call__(self, y, x=None):

        y = tf.gather(params=y, indices=self.indices, axis=-1)
        y = tf.reduce_sum(y, axis=-1)
        return tf.where(tf.equal(y,1), tf.ones_like(y), -1 * np.inf * tf.ones_like(y))

    def transformations(self):

        fs = []
        for m in self.binary_masks:
            fs.append((m,self.alsoNone))
        return fs


