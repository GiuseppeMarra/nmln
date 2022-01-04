import os, select, sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
import re
import warnings
import functools
from os.path import join
from sortedcontainers import SortedDict
from collections import OrderedDict


if os.name == 'nt':
    import msvcrt

def ReadFileByLines(file):
    try:
        with open(file, 'r') as f:
            return [line.rstrip() for line in f.readlines()]
    except IOError as e:
        assert False, "Couldn't open file (%s)" % file

def heardEnter():
    # Listen for the user pressing ENTER

    if os.name == 'nt':
        if msvcrt.kbhit():
            if msvcrt.getch() == b"q":
                print("Quit key pressed.")
                return True
        else:
            return False
    else:
        i, o, e = select.select([sys.stdin], [], [], 0.0001)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                return True
        return False

def save(path, o):
    with open(path, 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore(path):
    with open(path, 'rb') as handle:
        o = pickle.load(handle)
    return o


def ranking(relations_train, relations_test, relations_predicted):
    MRR = 0.0
    HITS1 = 0.0
    HITS3 = 0.0
    HITS5 = 0.0
    HITS10 = 0.0
    counter = 0.0

    for relation in relations_test.keys():
        r_test = relations_test[relation]
        r_predicted = relations_predicted[relation]
        r_train=relations_train[relation]
        n = np.shape(r_test)[0]
        for i in range(n):
            for j in range(n):
                if r_test[i, j] == 1:
                    for s, k in enumerate((i, j)):

                        # s k
                        # 0 i
                        # 1 j

                        predicted_score = r_predicted[i, j]

                        # we multiply for 1 - r_train in such a way to eliminate the scores coming from train data
                        if s == 0:
                            mask = (1 - r_train[i]) # ones row with 0 when the data is in the training data
                            mask[j] = 1
                            all_scores = sorted(r_predicted[i]*mask, reverse=True)
                        else:
                            mask = 1 - r_train[:,j]
                            mask[i] = 1
                            all_scores = sorted(r_predicted[:, j]*mask, reverse=True)
                        rank = all_scores.index(predicted_score) + 1

                        # if k == i:
                        #     all_scores = np.argsort(r_predicted[i])[::-1]
                        #     rank = list(all_scores).index(j) + 1
                        # else:
                        #     all_scores = np.argsort(r_predicted[:, j])[::-1]
                        #     rank = list(all_scores).index(i) + 1

                        counter += 1.0
                        if rank <= 1:
                            HITS1 += 1.0
                        if rank <= 3:
                            HITS3 += 1.0
                        if rank <= 5:
                            HITS5 += 1.0
                        if rank <= 10:
                            HITS10 += 1.0

                        MRR += 1.0 / rank

    MRR /= counter
    HITS1 /= counter
    HITS3 /= counter
    HITS5 /= counter
    HITS10 /= counter

    return (MRR, HITS1, HITS3, HITS5, HITS10)




class CustomDense(tf.keras.layers.Layer):


  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(CustomDense, self).__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = tf.dtypes.as_dtype(self.dtype)
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = tf.cast(inputs, self._compute_dtype)
      if tf.is_sparse(inputs):
        outputs = tf.sparse_tensor_dense_matmul(inputs, self.kernel)
      else:
        outputs = tf.math.mat_mul(inputs, self.kernel)
    if self.use_bias:
      outputs = outputs + self.bias
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs


def read_file_fixed_world(file, constants, predicates):
    from sortedcontainers import SortedDict

    t_constants = constants
    constants = SortedDict()
    for a in sorted(t_constants):
        constants[a] = len(constants)
    data = SortedDict()
    for p in predicates.keys():
        data[p] = []
    with open(file) as f:
        for line in f:
            p,rest = line.split("(")
            rest = re.sub(r"\)[\\*]*\s*", "", rest)
            args = rest.split(",")
            for i in range(len(args)):
                arg = args[i].replace(" ","")
                args[i] = arg
            try:
                if len(data[p])>0: assert(len(data[p][0]) == len(args))
            except:
                print(file)
            data[p].append(args)
    m_data = SortedDict()
    for key in sorted(data.keys()):
        k = key
        v = data[k]
        m_data[k] = np.zeros(shape=[len(constants) for _ in range(predicates[k])])
        for args in v:
            id = [constants[a] for a in args]
            np.put(m_data[k], np.ravel_multi_index(id, m_data[k].shape), 1)

    return constants,m_data,{}



def accuracy(y, targets):
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(targets, axis=1)),
                tf.float32)).numpy()


def binary_accuracy(y, targets):
    return tf.reduce_mean(
        tf.cast(tf.equal(y>0.5, targets>0.5),
                tf.float32)).numpy()


from bisect import bisect_left
from collections.abc import MutableMapping


class RangeBisection(MutableMapping):
    """Map ranges to values

    Lookups are done in O(logN) time. There are no limits set on the upper or
    lower bounds of the ranges, but ranges must not overlap.

    """
    def __init__(self, map=None):
        self._upper = []
        self._lower = []
        self._values = []
        if map is not None:
            self.update(map)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, point_or_range):
        if isinstance(point_or_range, tuple):
            low, high = point_or_range
            i = bisect_left(self._upper, high)
            point = low
        else:
            point = point_or_range
            i = bisect_left(self._upper, point)
        if i >= len(self._values) or self._lower[i] > point:
            raise IndexError(point_or_range)
        return self._values[i]

    def __setitem__(self, r, value):
        lower, upper = r
        i = bisect_left(self._upper, upper)
        if i < len(self._values) and self._lower[i] < upper:
            raise IndexError('No overlaps permitted')
        self._upper.insert(i, upper)
        self._lower.insert(i, lower)
        self._values.insert(i, value)

    def __delitem__(self, r):
        lower, upper = r
        i = bisect_left(self._upper, upper)
        if self._upper[i] != upper or self._lower[i] != lower:
            raise IndexError('Range not in map')
        del self._upper[i]
        del self._lower[i]
        del self._values[i]

    def __iter__(self):
        yield from zip(self._lower, self._upper)


def read_ontology_from_file(file):
    import problog
    program = problog.program.PrologFile(file)
    predicates = {}
    constants = set()
    for fact in program:
        p,arity = str(fact.functor), len(fact.args)
        if p not in predicates:
            predicates[p] = arity
        else:
            assert predicates[p] == arity, "Predicate {} arity inconsistency.".format(p)
        args = set([str(a.functor) for a in fact.args])
        if '-' in args:
            print()
        constants.update(args)
    return sorted(constants), predicates


# def parse_atom(atom):
#     import problog
#     # if atom[-1] != '.': atom= atom+'.'
#     program = problog.program.PrologString(atom)
#     return str(program[0].functor), [str(a.functor) for a in program[0].args]


from nmln.parser import atom_parser
parse_atom = atom_parser


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func



def generate_noise_coherent_with_transformations(transformations, pos_data, p_noise):
    if p_noise > 0:
        for m,_ in transformations:
            m2 = np.sum(m, 0)
            for k in range(len(pos_data)):
                if np.random.rand() < p_noise:
                    t = pos_data[k]
                    t = np.where(m2 > 0, np.zeros_like(t), t)
                    pos_data[k] = t + m[np.random.choice(len(m), size=1)][0]
    return pos_data




def __read_ntp_ontology_only(base_path, file):
    file = join(base_path,file)
    predicates = OrderedDict()
    constants = OrderedDict()
    with open(file) as f:
        for line in f:
            p,rest = line.split("(")
            args = rest.split(")")[0].split(",")
            for i in range(len(args)):
                arg = args[i].replace(" ","")
                args[i] = arg
                if arg not in constants:
                    constants[arg] = len(constants)
            if p not in predicates:
                predicates[p]=len(predicates)
    return constants,predicates







def __read_ntp_file_fixed_world(base_path, file, constants, predicates):
    file = join(base_path,file)
    data = SortedDict()
    for p in predicates.keys():
        data[p] = []
    with open(file) as f:
        for line in f:
            p, rest = line.split("(")
            args = rest.split(")")[0].split(",")
            for i in range(len(args)):
                arg = args[i].replace(" ","")
                args[i] = arg
            try:
                if len(data[p])>0: assert(len(data[p][0]) == len(args))
            except:
                print(file)
            data[p].append(args)
    m_data = SortedDict()
    for key in sorted(data.keys()):
        k = key
        v = data[k]
        m_data[k] = np.zeros(shape=[len(constants) for _ in range(len(data[k][0]))])
        for args in v:
            id = [constants[a] for a in args]
            np.put(m_data[k], np.ravel_multi_index(id, m_data[k].shape), 1)

    return constants,m_data,{}

def ntp_dataset(name, base_path):

    ground_path = join(name, name+".nl")
    train_path = join(name, "train.nl")
    valid_path = join(name, "dev.nl")
    test_path = join(name, "test.nl")



    constants, predicates = __read_ntp_ontology_only(base_path, ground_path)
    _, ground_relations, _ = __read_ntp_file_fixed_world(base_path, ground_path, constants, predicates)
    _, train_relations, _ = __read_ntp_file_fixed_world(base_path, train_path, constants, predicates)
    _, valid_relations, _ = __read_ntp_file_fixed_world(base_path, valid_path, constants, predicates)
    _, test_relations, _ = __read_ntp_file_fixed_world(base_path, test_path, constants, predicates)


    def to_linear(relations):
        return np.reshape(np.concatenate([np.reshape(t, [-1]) for k,t in relations.items()], axis=0),[1, -1])

    return constants, predicates, to_linear(ground_relations), to_linear(train_relations), valid_relations, test_relations


def __read_ntp_file_triple(base_path, file, constants, predicates):
    file = join(base_path,file)

    # We initialize a dictionary <PredicateName, List_of_Tuples_of_Constants_keys>
    data = []


    with open(file) as f:
        for line in f:
            p, rest = line.split("(")
            args = rest.split(")")[0].split(",")
            for i in range(len(args)):
                arg = args[i].replace(" ","")
                arg = constants[arg]
                args[i] = arg
            data.append((args[0], predicates[p], args[1]))


    return data


def ntp_dataset_triple(name, base_path):

    ground_path = join(name, name+".nl")
    train_path = join(name, "train.nl")
    valid_path = join(name, "dev.nl")
    test_path = join(name, "test.nl")



    constants, predicates = __read_ntp_ontology_only(base_path,train_path)
    ground_facts = __read_ntp_file_triple(base_path,ground_path, constants, predicates)
    train_facts = __read_ntp_file_triple(base_path,train_path, constants, predicates)
    valid_facts = __read_ntp_file_triple(base_path,valid_path, constants, predicates)
    test_facts = __read_ntp_file_triple(base_path,test_path, constants, predicates)


    return constants, predicates, ground_facts, train_facts, valid_facts, test_facts



def ntn_dataset_triple_we(dataset, base_path):

    if dataset == "wn":
        name = "WordnetNTN"
    elif dataset == "fb":
        name = "FreebaseNTN"
    else:
        raise Exception("Dataset unknown for ntn_dataset_triple")

    data_path = join(base_path, name)

    def __read_ontology():
        constants_file_path = join(data_path, "entities.txt")
        relations_file_path = join(data_path, "relations.txt")
        predicates = OrderedDict()
        constants = OrderedDict()

        # Constants
        with open(constants_file_path) as f:
            for line in f:
                arg = line.replace("\n", "")
                if arg not in constants:
                    constants[arg] = len(constants)

        with open(relations_file_path) as f:
            for line in f:
                arg = line.replace("\n", "")
                if arg not in predicates:
                        predicates[arg] = len(predicates)

        return constants, predicates


    def __inner__(split_path, constants, predicates, is_eval=False):
        triples = []
        if is_eval:
            corrupted_triples = []
        with open(split_path) as f:
            for line in f:
                splits = line.split("\t")
                splits = [s.strip() for s in splits]
                triple = (constants[splits[0]], predicates[splits[1]], constants[splits[2]])
                if is_eval and "-1" in splits[3]:
                    corrupted_triples.append(triple)
                else:
                    triples.append(triple)

        if is_eval:
            triples = triples + corrupted_triples
        return triples

    def load_embeddings(embed_path, constants):

        embed_dict = sio.loadmat(embed_path)
        words = embed_dict["words"][0]
        we = embed_dict["We"].T

        words_to_idx = {}
        WE = []

        PAD = "<<PAD>>"
        words = np.concatenate(([PAD], words), axis=0)
        we = np.concatenate((np.zeros([1, len(we[0])]), we), axis=0)
        words_to_idx[PAD] = 0
        WE.append(we[0])

        UNK = "*unknown*"

        for j in range(1, len(words)):
            w = words[j]
            if w[0] not in words_to_idx:
                words_to_idx[w[0]] = len(words_to_idx)
                WE.append(we[j])



        num_words = len(words_to_idx)
        constants_embeddings = []
        constants_bow = []

        max_l = 0
        for jh, c in enumerate(constants):
            ws = c.replace("__","").split("_")
            if len(ws) > 1:
                ws = ws[:-1]
            e = []
            max_l = max(max_l, len(ws))
            constants_bow.append([])
            for w in ws:
                w = w if w in words_to_idx else UNK
                e.append(WE[words_to_idx[w]])
                constants_bow[-1].append(words_to_idx[w])


            constants_embeddings.append(np.mean(e, axis=0))

        for i in range(len(constants_bow)):
            constants_bow[i] = np.array(constants_bow[i] + [words_to_idx[PAD] for _ in range(max_l - len(constants_bow[i]))])

        constants_bow = np.array(constants_bow)
        return np.array(constants_embeddings).astype(np.float32) , constants_bow, np.array(WE).astype(np.float32)
        # return np.array(constants_embeddings).astype(np.float32), None, None

    train_path = join(data_path, "train.txt")
    valid_path = join(data_path, "dev.txt")
    test_path = join(data_path, "test.txt")
    embed_path = join(data_path, "initEmbed")



    constants, predicates = __read_ontology()
    train_facts = __inner__(train_path, constants, predicates)
    valid_facts = __inner__(valid_path, constants, predicates, is_eval=True)
    test_facts = __inner__(test_path, constants, predicates, is_eval=True)

    embeddings, bow, we = load_embeddings(embed_path, constants)
    # embeddings, bow, we = None, None, None

    ground_facts = train_facts + valid_facts[:len(valid_facts)//2] + test_facts[:len(test_facts)//2]


    return constants, predicates, ground_facts, train_facts, valid_facts, test_facts, embeddings, bow, we


