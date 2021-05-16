from pyparsing import *
ParserElement.enablePackrat()
from collections import OrderedDict
from itertools import product
import numpy as np
from nmln.logic import *
import itertools

class Node():

    def __init__(self):
        self.name = 'node'
        self.args = []


class Variable(Node):

    def __init__(self, name, domain):
        super().__init__()
        self.name = name
        self.domain = domain

    def set_indices(self, indices):
        self.indices = indices

class Atom(Node):

    def __init__(self, predicate, args, idx):
        super().__init__()
        self.predicate = predicate
        self.name = predicate.name
        self.args = args
        self.idx = idx

    def set_indices(self, offset_range):
        base = offset_range
        for i,v in enumerate(self.args):
            next_domain_size = self.args[i+1].domain.num_constants if i<(len(self.args)-1) else 1
            base = base + v.indices * next_domain_size
        self.indices = base
        self.num_groundings = len(base)

    def ground(self, herbrand_interpretation, formula_filter=None):
        ground_indices = self.indices
        if formula_filter:
            groundings = formula_filter.ground(herbrand_interpretation)
            filter_indices = np.squeeze(formula_filter.compile(
                groundings, BooleanLogic).numpy())
            ground_indices = self.indices[filter_indices>0]


        if ground_indices is None:
            raise Exception("Atom indices not set")

        if isinstance(herbrand_interpretation, np.ndarray):
            e = np.expand_dims(np.take(herbrand_interpretation, ground_indices, axis=-1), axis=-1)
            return e
        # TODO(giuseppe): This is the only point linked to tensorflow.
        # If we want to make the compialtion dynamic we just provide the
        # compilation function as parameter of the atom and to the
        # constraint in turn.
        return tf.expand_dims(
            tf.gather(herbrand_interpretation, ground_indices, axis=-1),
            axis=-1)

    def compile(self, groundings):
        n = len(groundings.shape)
        start = np.zeros([n], dtype=np.int32)
        size = -1*np.ones([n], dtype=np.int32)
        start[-1] = int(self.idx)
        size[-1]=1
        sl =  tf.squeeze(tf.slice(groundings, start, size), axis=-1)
        # sl2 = groundings[:, :, :, ontology.idx]
        return sl



class Operator(Node):

    def __init__(self, f, args, name):
        super().__init__()
        self.f = f
        self.name = name
        self.args = args

    def compile(self, groundings):
        targs = []
        for a in self.args:
            cc = a.compile(groundings)
            targs.append(cc)
        return self.f(targs)


def all_combinations(n):
    l = list(itertools.product([False, True], repeat=n))
    return np.array(l).astype(np.float32)

def all_combinations_in_position(n,j):
    base = np.concatenate((np.zeros([2**(n-1)]), np.ones([2**(n-1)])),axis=0)
    base_r = np.reshape(base, [2 for _ in range(n)])

    transposition = list(range(n))
    for i in range(j):
        k = transposition[i]
        transposition[i]=transposition[i+1]
        transposition[i + 1] = k
    base_t = np.transpose(base_r, transposition)
    base_f = np.reshape(base_t, [-1])
    return base_f

class Formula(object):

    def __init__(self, ontology, definition, variables=None):
        self.ontology = ontology
        self.variables = OrderedDict()
        self.atoms = []
        self.logic = None
        if variables is not None:
            self.variables = variables
        self.expression_tree = self.parse(definition)
        self.definition = definition

        # Computing Variable indices
        if variables is  None:
            sizes = []
            for i, (k, v) in enumerate(self.variables.items()):
                sizes.append(range(v.domain.num_constants))

            # Cartesian Product
            indices = [i for i in product(*sizes)] #TODO consider using tiling + stacking to see if it improves performances

            indices = np.array(indices)
            for i, (k, v) in enumerate(self.variables.items()):
                v.set_indices(indices[:,i])

        #Computing Atom indices
        for i, a in enumerate(self.atoms):
            a.set_indices(self.ontology._predicate_range[a.predicate.name][0])

        #Num groundings of the formula = num grounding of a generic atom
        self.num_groundings = self.atoms[0].num_groundings

        self.num_given = sum([1 for a in self.atoms if a.predicate.given])


    def arity(self):
        if self.variables is not None:
            return len(self.variables)
        return 0

    def num_atoms(self):
        return len(self.atoms)

    # neighborOf(x, z) and locatedIn(z, y) -> locatedIn(x, y)
    # neighborOf(x, z)
    def grounding_indices(self, filter=None, kb=None):
        indices = [a.indices for a in self.atoms]
        indices = np.vstack(indices).T
        if filter is not None:
            formula_filter = Formula(
                self.ontology, filter, variables=self.variables)
            groundings = formula_filter.ground(kb)
            filter = np.squeeze(formula_filter.compile(
                groundings, BooleanLogic).numpy())
            indices = indices[filter>0]

        # Computes a table atom_index -> connected_atom_indices
        #ontology.atom2connected_atoms = {}
        #for i in range(indices.shape[0]):
        #    for j in range(indices.shape[1]):
        #        for k in range(indices.shape[1]):
        #            if j != k:
        #                if indices[i][j] not in ontology.atom2connected_atoms:
        #                    ontology.atom2connected_atoms[indices[i][j]] = []
        #                ontology.atom2connected_atoms[indices[i][j]].append(
        #                    indices[i][k])
        return indices

    def all_assignments_to_a_grounding(self):
        #this corresponds to 1 sample, 1 grounding, 2^n possible assignments, n values of a single assignment [1,1, 2^n, n]
        n = len(self.atoms)
        l = list(itertools.product([True, False], repeat=n))
        return np.array(l)

    def all_sample_groundings_given_evidence(self, evidence, evidence_mask):

        y_e = self.ground(herbrand_interpretation=evidence)
        y_e = tf.squeeze(y_e, axis=-2) # removing the grounding assignments dimension since we will add it here
        m_e = self.ground(herbrand_interpretation=evidence_mask)
        m_e = tf.squeeze(m_e, axis=-2) # removing the grounding assignments dimension since we will add it here


        n_examples = len(y_e)
        n_groundings = len(y_e[0])
        n_variables = len(self.atoms)
        k = n_variables - self.num_given
        n_assignments = 2**k

        shape = [n_variables, n_examples, n_groundings, 2 ** k]

        indices = tf.where(m_e[0][0] > 0)
        given = tf.gather(y_e, tf.reshape(indices,[-1]), axis=-1)
        given = tf.transpose(given, [2,1,0])
        given = tf.reshape(given, [self.num_given, n_examples, n_groundings, 1])
        given = tf.cast(tf.tile(given, [1, 1, 1, 2 ** k]), tf.float32)
        first = tf.scatter_nd(shape=shape, indices=indices, updates=given)

        indices = tf.where(m_e[0][0]  < 1)
        l = list(product([False, True], repeat=k))
        comb = np.stack(l, axis=1).astype(np.float32)
        assignments = np.tile(np.reshape(comb, [-1, 1, 1, 2 ** k]), [1, n_examples, n_groundings, 1])
        second = tf.scatter_nd(shape=shape, indices=indices, updates=assignments)

        final = tf.transpose(first + second, [1, 2, 3, 0])
        return final


    def all_sample_groundings_given_evidencev2(self, evidence, evidence_mask):

        pass

    def _create_or_get_variable(self, id, domain):
        if id in self.variables:
            assert self.variables[id].domain == domain, "Inconsistent domains for variables and predicates"
        else:
            v = Variable(id, domain)
            self.variables[id] = v
        return self.variables[id]



    def _parse_action(self, class_name):
        def _create(tokens):

            if class_name == "Atomic":
                predicate_name = tokens[0]
                predicate = self.ontology.predicates[predicate_name]
                args = []
                for i, t in enumerate(tokens[1:]):
                    args.append(self._create_or_get_variable(
                        t, predicate.domains[i]))
                a = Atom(predicate, args, len(self.atoms))
                self.atoms.append(a)
                return a
            elif class_name == "NOT":
                args = tokens[0][1:]
                return Operator(lambda x: self.logic._not(x), args, name = "NOT")
            elif class_name == "AND":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._and(x), args, name = "AND")
            elif class_name == "OR":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._or(x), args, name = "OR")
            elif class_name == "XOR":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._xor(x), args, name = "XOR")
            elif class_name == "IMPLIES":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._implies(x), args, name = "IMPLIES")
            elif class_name == "IFF":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._iff(x), args, name = "IFF")

        return _create

    def parse(self, definition):

        left_parenthesis, right_parenthesis, colon, left_square, right_square = map(Suppress, "():[]")
        symbol = Word(alphas)

        ''' TERMS '''
        var = symbol
        # var.setParseAction(ontology._createParseAction("Variable"))

        ''' FORMULAS '''
        formula = Forward()
        not_ = Keyword("not")
        and_ = Keyword("and")
        or_ = Keyword("or")
        xor = Keyword("xor")
        implies = Keyword("->")
        iff = Keyword("<->")

        forall = Keyword("forall")
        exists = Keyword("exists")
        forall_expression = forall + symbol + colon + Group(formula)
        forall_expression.setParseAction(self._parse_action("FORALL"))
        exists_expression = exists + symbol + colon + Group(formula)
        exists_expression.setParseAction(self._parse_action("EXISTS"))

        relation = oneOf(list(self.ontology.predicates.keys()))
        atomic_formula = relation + left_parenthesis + delimitedList(var) + right_parenthesis
        atomic_formula.setParseAction(self._parse_action("Atomic"))
        espression = forall_expression | exists_expression | atomic_formula
        formula << infixNotation(espression, [
            (not_, 1, opAssoc.RIGHT,self._parse_action("NOT")),
            (and_, 2, opAssoc.LEFT, self._parse_action("AND")),
            (or_, 2, opAssoc.LEFT, self._parse_action("OR")),
            (xor, 2, opAssoc.LEFT, self._parse_action("XOR")),
            (implies, 2, opAssoc.RIGHT, self._parse_action("IMPLIES")),
            (iff, 2, opAssoc.RIGHT, self._parse_action("IFF"))
        ])

        constraint = var ^ formula
        tree = constraint.parseString(definition, parseAll=True)
        return tree[0]

    def compile(self, groundings, logic=BooleanLogic):
        self.logic = logic
        t = self.expression_tree.compile(groundings)
        self.logic = None
        return t

    def ground(self, herbrand_interpretation, filter=None):
        if filter is not None:
            formula_filter = Formula(
                self.ontology, filter, variables=self.variables)
            groundings_filter = formula_filter.ground(herbrand_interpretation)
            filter = np.squeeze(formula_filter.compile(
                groundings_filter, BooleanLogic).numpy())

        if isinstance(herbrand_interpretation, np.ndarray):
            groundings =  np.stack([a.ground(herbrand_interpretation) for a in self.atoms], axis=-1)
        else:
            groundings = tf.stack([a.ground(herbrand_interpretation) for a in self.atoms], axis=-1)


        if filter is not None:
            return groundings[filter>0] #TODO check if the filter works with tf
        else:
            return groundings

##########################################
# Get a list of cliques from the formulas.
def GetCliqueIndices(formula, num_atoms, positional,
                     filter=None, hb=None):
    num_entries = num_atoms
    if filter is not None:
        assert hb is not None
    # The cliques for one formula.
    cliques = formula.grounding_indices(filter=filter, kb=hb)
    num_cliques = len(cliques)
    assert num_cliques > 0
    clique_size = len(cliques[0])
    print("Number of edges/cliques from formula", formula.definition, "=",
          len(cliques))
    cliques_indices = np.expand_dims(np.arange(
        num_entries, num_entries + len(cliques)), axis=1)
    cliques_indices = np.broadcast_to(cliques_indices, [num_cliques, clique_size])
    indices = np.stack((cliques_indices, cliques), axis=2)
    indices = np.reshape(indices, [-1, 2])
    if not positional:
        values = np.ones([num_cliques*clique_size], dtype=np.float32)
    else:
        values = np.expand_dims(np.arange(clique_size), 0)
        values = np.tile(values, [num_cliques, 1])
        values = np.reshape(values, [-1])
    return cliques, indices, values, num_cliques



symbol = Word(alphanums+'_')
left_parenthesis, right_parenthesis, colon, left_square, right_square, dot = map(Suppress, "():[].")
parser_atom = symbol + left_parenthesis + delimitedList(symbol) + right_parenthesis + Optional(dot)

def atom_parser(atom_string):
    tokens = parser_atom.parseString(atom_string)
    return tokens[0], tokens[1:]

