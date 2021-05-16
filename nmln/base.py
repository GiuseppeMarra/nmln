import nmln.utils as utils
from nmln.utils import deprecated
from nmln.parser import Formula
import numpy as np
from collections.abc import Iterable
from collections import OrderedDict
from itertools import permutations, product
import warnings


class Domain():

    def __init__(self, name, num_constants, constants=None, features: np.array = None):
        """
        Class of objects representing a Domain of domains in a FOL theory

        Args:
            name: name of the domain
            num_constants: number of the constant
            constants: list of domains identifiers (strings or integers)
            features: np.array, a matrix [num_constants, feature_size], where each row i represents the features of
                      constant i
        """
        if name is not None:
            self.name = str(name)
        else:
            raise Exception("Attribute 'name' is None.")
        self.num_constants = num_constants
        self.constants = constants

        """Map domains names to id (row index)"""
        self.constant_name_to_id = {c:i for i,c in enumerate(self.constants)}
        if constants is not None:
            assert self.num_constants == len(constants)
        if features is not None:
            assert features.shape[0] == len(constants)
            self.features = features
        else:
            self.features = np.expand_dims(np.eye(num_constants, dtype=np.float32), 0)

    def __hash__(self):
        return str(self.name).__hash__()



class Predicate():
    def __init__(self, name, domains, given=False):
        """
        Class of objects representing a relations in a FOL theory.

        Args:
            name: the unique name of the predicate
            domains: list(nmln.Domain), a positional list of domains
        """
        self.name = name
        self.given = given

        self.domains = []
        groundings_number = 1
        for domain in domains:
            if not isinstance(domain, Domain):
                raise Exception(str(domain) + " is not an instance of " + str(Domain))
            self.domains.append(domain)
            groundings_number *= domain.num_constants
        self.groundings_number = groundings_number
        self.arity = len(self.domains)

    def __lt__(self, other):
        return self.name < other.name


class Ontology():
    def __init__(self, domains, predicates):
        """
        The central object. It represents a multi-sorted FOL language.

        Args:
            domains: the domains of the language
            predicates: the predicates of the language
        """
        self.domains = {}
        self._domain_list = []
        self.predicates = OrderedDict()
        self.herbrand_base_size = 0
        self._predicate_range = OrderedDict()
        self._range_to_predicate = utils.RangeBisection()
        self.finalized = False
        self.constraints = []
        # Overall, number of elements in the assignment vector.
        self._linear_size = 0

        for d in domains:
            self.__add_domain(d)

        if len(domains) == 1:
            self.num_constants = domains[0].num_constants

        self.tuple_indices = {}
        for p in predicates:
            self.__add_predicate(p)

        self.__create_indexing_scheme()

        """ For some datasets, computing the indices of fragments is heavy. We store them."""
        self.all_fragments_cache = {}


    def __str__(self):
        s = ""
        s += "Domains (%d): "% len(self.domains) + ", ".join(["%s (%d)" % (name, domain.num_constants) for name, domain in self.domains.items()]) + "\n"
        s += "Predicates (%d):" % len(self.predicates) + ", ".join(self.predicates.keys()) + "\n"
        return s


    @staticmethod
    def from_file(file):
        """
        Factory method. Instantiate a new Ontology by reading from a file

        Args:
            file: the file containing all the facts to read the ontology from

        Returns: the Ontology object

        """
        constants, predicates = utils.read_ontology_from_file(file)
        d = Domain(name="domain", constants=constants, num_constants=len(constants))
        predicates = [Predicate(p, domains=[d for _ in range(a)]) for p, a in predicates.items()]
        return Ontology(domains=[d], predicates=predicates)

    def __check_multidomain(self):
        """
            Internal function to check if the FOL language is multi-sorted (i.e. multiple domains)
        """
        if len(self.domains)>1:
            raise Exception("This operation does not allow multi domains")

    def __add_domain(self, d):
        if not isinstance(d, Iterable):
            D = [d]
        else:
            D = d
        for d in D:
            if d.name in self.domains:
                raise Exception("Domain %s already exists" % d.name)
            self.domains[d.name] = d
            self._domain_list.append(d)

    def __add_predicate(self, p):
        if not isinstance(p, Iterable):
            P = [p]
        else:
            P = p
        for p in P:
            if p.name in self.predicates:
                raise Exception("Predicate %s already exists" % p.name)
            self.predicates[p.name] = p
            self._predicate_range[p.name] = (self.herbrand_base_size, self.herbrand_base_size + p.groundings_number)
            self._range_to_predicate[(self.herbrand_base_size, self.herbrand_base_size + p.groundings_number - 1)] = p.name
            self.herbrand_base_size += p.groundings_number
            k = tuple([d.name for d in p.domains])
            if k not in self.tuple_indices:
                # Cartesian product of the domains.
                ids = np.array([i for i in product(*[range(self.domains[d].num_constants) for d in k])])
                self.tuple_indices[k] = ids

    def __create_indexing_scheme(self):
        """
            Creates the indexing scheme used by the Ontology object for all the logic to tensor operations.
        """
        # Managing a linearized version of this logic
        self._up_to_idx = 0  # linear max indices
        self._dict_indices = {}  # mapping potentials id to correspondent multidimensional indices tensor

        self.finalized = False
        self._linear = None
        self._linear_evidence = None

        # Overall, number of elements in the assignment vector.
        self._linear_size = 0
        for p in self.predicates.values():
            # For unary predicates, this is just the domain size as [size]
            # For n-ary predicates, this is just the tensor of domain sizes [d1_size, d2_size, ...]
            shape = [d.num_constants for d in p.domains]
            # Overall domain size.
            predicate_domain_size = np.prod(shape)
            start_idx = self._up_to_idx
            end_idx = start_idx + predicate_domain_size
            self._up_to_idx = end_idx
            # print('Dict Indices', start_idx, end_idx)
            self._dict_indices[p.name] = np.reshape(np.arange(start_idx, end_idx), shape)
            self._linear_size += predicate_domain_size
        self.finalized=True

    def get_constraint(self, formula):
        """
        Create a nmln.Formula using this ontology for the parse

        Args:
            formula: a string containing the formula definition

        Returns: an nmln.Formula object

        """
        return Formula(self, formula)

    @deprecated
    def FOL2LinearState(self, file):
        return self.file_to_linearState(file)


    def file_to_linearState(self, file):
        state = np.zeros(self.linear_size())
        ids = []
        with open(file) as f:
            for line in f:
                ids.append(self.atom_string_to_id(line))
        state[ids] = 1
        return state


    def mask_by_atom_strings(self, atom_strings):
        """
            Creates a linear mask starting from a list of atom strings of interest.

        Args:
            atom_strings: list of atom strings

        Returns:

        """
        mask = np.zeros(self.linear_size())
        ids = []
        for atom in atom_strings:
            ids.append(self.atom_string_to_id(atom))
        mask[ids] = 1
        return mask


    def mask_by_constant(self, constants, negate=False):
        """
            Creates a linear mask from a list of domains of interest.

        Args:
            constants: list of domains ids
            negate: If negate is true, the mask has ones when not matching the domains.

        Returns:

            a mask, i.e. a [self.linear_size()] shaped np.array

        """
        constant_set = frozenset(constants)
        if negate is False:
            mask = np.zeros(self.linear_size())
            non_default_value = 1
        else:
            mask = np.ones(self.linear_size())
            non_default_value = 0

        for i in range(self.linear_size()):
            data = self.id_to_predicate_constant_strings(i)
            atom_constants = data[1:]
            for c in atom_constants:
                if c in constant_set:
                    mask[i] = non_default_value
                    break
        return mask

    def mask_by_constant_and_predicate(self, constants, predicates,
                                       negate_constants=False):
        """

         Creates a linear mask from a list of domains and predicates of interest.


        Args:
            constants: list of domains ids
            predicates: list of nmln.Predicates
            negate_constants:

        Returns: the mask, i.e. a [self.linear_size()] shaped np.array

        """
        if predicates is None or not predicates:
            return self.mask_by_constant(constants, negate=negate_constants)

        constant_set = frozenset(constants)
        mask = np.zeros(self.linear_size())

        if negate_constants is False:
            non_default_value = 1
        else:
            non_default_value = 0

        for p in predicates:
            a, b = self._predicate_range[p]
            if negate_constants is True:
                mask[a:b] = 1
            for i in range(a, b):
                data = self.id_to_predicate_constant_strings(i)
                atom_constants = data[1:]
                for c in atom_constants:
                    if c in constant_set:
                        mask[i] = non_default_value
                        break
        return mask

    def ids_by_constant_and_predicate(self, constants, predicates):
        """

            Get all ids for a predicate of entries containing one of the domains.

        Args:
            constants:
            predicates:

        Returns:
            list of atom ids containing one of the domains

        """
        ids = []
        constant_set = frozenset(constants)
        for p in predicates:
            a, b = self._predicate_range[p]
            for i in range(a, b):
                data = self.id_to_predicate_constant_strings(i)
                atom_constants = data[1:]
                for c in atom_constants:
                    if c in constant_set:
                        ids.append(i)
                        break
        return ids

    def id_range_by_predicate(self, name):
        """
           Return (a,b) tuple, where [a,b[ is the interval of indices of atoms of predicate "name"
           in the linearized indexing.

        Args:
            name: predicate name

        Returns:
            a tuple (a,b) of indices

        """
        a, b = self._predicate_range[name]
        assert b - a > 0, 'Can not find predicate %s' % name
        return a, b



    @deprecated
    def file_to_linearState_old(self, file):
        """Read a prolog-like file using the current ontology to parse it. Returns a linear state"""
        self.__check_multidomain()
        pp = OrderedDict({p.name: p.arity for p in self.predicates.values()})
        constants, predicates, evidences = utils.read_file_fixed_world(file, list(self.domains.values())[0].domains, pp)
        linear = np.zeros(shape=self.linear_size())
        for p,v in predicates.items():
            linear[self._dict_indices[p]] = v
        return linear

    @deprecated
    def linear2Dict(self, linear_state):
        return self.linear_to_fol_dictionary(linear_state)

    def linear_to_fol_dictionary(self, linear_state):
        """
            Create a dictionary mapping predicate names to np.array. For each key-value pair, the "value" of the
            dictionary array is the adiacency matrix of the predicate with name "key".

        Args:
            linear_state: a np.array with shape [self.linear_size()]

        Returns:
            a dictionary mapping predicate names to np.array

        """
        d = OrderedDict()
        for p in self.predicates.values():
            d[p.name] = np.take(linear_state, self._dict_indices[p.name])
        return d

    def fol_dictionary_to_linear(self, dictionary):
        """

            Gets an input dictionary, mapping names to np.array. Return a concatenated linear version of all the values
            of the dictionary. This function is the inverse of Ontology.linear_to_fol_dictionary.

        Args:
            dictionary: a dictionary mapping predicate names to np.array

        Returns:
            a np.array

        """
        hb = np.zeros([self.linear_size()])
        for name in self.predicates:
            if name not in dictionary:
                raise Exception("%s predicate array is not provided in the dictionary." % name)
            array = dictionary[name]
            a, b = self._predicate_range[name]
            try:
                hb[a:b]=np.reshape(array, [-1])
            except:
                array = array.todense()
                hb[a:b]=np.reshape(array, [-1])
        return np.reshape(hb, [1, self.linear_size()])


    def fol_dictionary_to_linear_tf(self, dictionary, axis=1):
        """

            Return a concatenation of the keys of the dictionary along a specified "axis.

        Args:
            dictionary: a dictionary mapping predicate names to arrays
            axis: the axis of the concatenation. Defaults to 1.

        Returns:
            a concatenation of the keys of the dictionary along a specified "axis". The order of the concatenation is the
            iteration order of Ontology.predicates.

        """
        import tensorflow as tf
        # todo merge with the previous (numpy) one in some way.
        hb = []
        for name in self.predicates:
            if name not in dictionary:
                raise Exception("%s predicate array not provided in the dictionary." % name)
            array = dictionary[name]
            hb.append(array)
        res =  tf.concat(hb, axis=axis)
        # assert res.shape[0] == ontology.linear_size()
        return res


    def id_to_atom(self, id_atom):
        predicate_name = self._range_to_predicate[id_atom]
        shape = self._dict_indices[predicate_name].shape
        ids = np.unravel_index(id_atom - self._predicate_range[predicate_name][0], shape)
        return predicate_name, ids

    def id_to_atom_string(self, id_atom):
        p_name, cs = self.id_to_atom(id_atom)
        p = self.predicates[p_name]
        return p_name + "(%s)" % ",".join([p.domains[i].domains[c] for i, c in enumerate(cs)])

    def id_to_predicate_constant_strings(self, id_atom):
        p_name, cs = self.id_to_atom(id_atom)
        p = self.predicates[p_name]
        return [p_name] + [p.domains[i].domains[c] for i, c in enumerate(cs)]

    def atom_string_to_id(self, atom):
        predicate, constants = utils.parse_atom(atom)
        p = self.predicates[predicate]
        constants_ids = tuple(p.domains[i].constant_name_to_id[c] for i,c in enumerate(constants))
        return self.atom_to_id(predicate, constants_ids)

    def atom_to_id(self, predicate_name, constant_ids):
        return self._dict_indices[predicate_name][tuple(constant_ids)]

    def linear_size(self):
        return self._linear_size

    def sample_fragments_idx(self, k, num=100, get_ids = False):
        self.__check_multidomain()
        ii = []
        all_ids = []
        for _ in range(num):
            i=[]
            num_constants = list(self.domains.values())[0].num_constants
            idx = np.random.choice(num_constants, size=k, replace=False)
            idx = np.random.permutation(idx)
            all_ids.append(idx)
            for p in self.predicates.values():
                a = p.arity
                f_idx = self._dict_indices[p.name]
                for j in range(a):
                    f_idx = np.take(f_idx, idx, axis=j)
                f_idx = np.reshape(f_idx, [-1])
                i.extend(f_idx)
            ii.append(i)
        res = np.stack(ii, axis=0)
        if not get_ids:
            return res
        else:
            return res, np.stack(all_ids, axis=0)


    def all_fragments_idx_wrong(self, k, get_ids = False, get_atom_to_fragments_mask=False):
        self.__check_multidomain()
        if k in self.all_fragments_cache is not None:
            groundings_hb_indices, indices = self.all_fragments_cache[k]
        else:
            num_constants = list(self.domains.values())[0].num_constants
            indices = np.array(list(permutations(range(num_constants), r=k)))
            groundings_hb_indices = []
            for i, (name, predicate) in enumerate(self.predicates.items()):
                predicate_range = self._predicate_range[name]
                size = predicate.domains[0].num_constants
                for j in range(k):
                    groundings_hb_indices.append(predicate_range[0] + size * indices[:, j:j + 1] + indices)

            groundings_hb_indices = np.concatenate(groundings_hb_indices, axis=1)
            self.all_fragments_cache[k] = groundings_hb_indices, indices
        if get_ids:
            to_return = groundings_hb_indices, indices
        else:
            to_return = groundings_hb_indices
        return to_return

    def all_fragments_idx(self, k, get_ids=False, get_atom_to_fragments_mask=False):

        ii = []
        all_ids = []
        num_constants = list(self.domains.values())[0].num_constants
        for idx in permutations(range(num_constants), k):
            all_ids.append(idx)
            i = []
            for p in self.predicates.values():
                a = p.arity
                f_idx = self._dict_indices[p.name]
                for j in range(a):
                    f_idx = np.take(f_idx, idx, axis=j)
                f_idx = np.reshape(f_idx, [-1])
                i.extend(f_idx)
            ii.append(i)
        res = np.stack(ii, axis=0)
        to_return = res
        if get_ids:
            to_return = [res,np.stack(all_ids, axis=0)]

        if get_atom_to_fragments_mask:
            atom_to_fragments_mask = np.zeros([self.linear_size(), len(res)])
            for i in range(len(res)):
                for j in range(len(res[0])):
                    atom_id = res[i,j]
                    atom_to_fragments_mask[atom_id, i] = 1
            to_return = to_return+[atom_to_fragments_mask]

        return to_return


    def one_factors(self, k=3, return_pairs=False):

        if self.num_constants % 2 ==0:
            n = self.num_constants
            odd = False
        else:
            n = self.num_constants + 1
            odd = True

        """Creating the indices for the one-factors"""
        A = []
        r = np.arange(n//2)
        r2 = np.arange(n//2, n-1)
        r2 = r2[::-1]
        for i in range(n - 1):
            rr = np.mod(r + i, n-1)
            rr2 = np.mod(r2 + i, n-1)
            rr2 = np.concatenate(([n-1],rr2), axis=0)
            a = np.stack((rr, rr2), axis=1)
            A.append(a)
        A = np.stack(A, axis=0)

        """Now I create a map between a pair of indices in a factorization and its correspondent k-factor """
        idx, ids = self.all_fragments_idx(k, get_ids=True)
        d = OrderedDict()
        for j, id in enumerate(ids):
            for l,k in permutations(id, 2):
                if (l,k) not in d:
                    d[(l, k)] = []
                d[(l,k)].append(j)

        # Now I create a vector like the map
        B = []
        for of in A:
            C = []
            for a,b in of:
                    C.append(d[tuple([a,b])])
            B.append(C)
        B = np.array(B)



        """Now I need the indices of the interpretations of each of the factorizations"""
        idx, ids = self.all_fragments_idx(2, get_ids=True)
        d = OrderedDict()
        for j, (l,k) in enumerate(ids):
            d[(l, k)] = idx[j]

        # Now I create a vector like the map
        D = []
        for of in A:
            C = []
            for a,b in of:
                    C.append(d[tuple([a,b])])
            D.append(C)
        D = np.array(D)



        if not return_pairs:
            return D, B
        else:
            return A,D,B




    def size_of_fragment_state(self, k):
        self.__check_multidomain()
        size = 0
        for p in self.predicates.values():
            size += k**p.arity
        return size
