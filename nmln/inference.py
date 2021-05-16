import tensorflow as tf
import abc
import numpy as np
import time
import nmln.potentials as potentials
import nmln.logic as logic

eps = 1e-12

class Inference():
    def __init__(self, global_potential, parameters=None):
        self.global_potential = global_potential
        self.parameters = parameters

    @abc.abstractmethod
    def infer(self, x=None, y=None):
        pass



class Sampler(object):

    @tf.function
    def sample(self, conditional_data=None, num_samples=None, minibatch = None):
        pass


class ProposalDistribution():

    def sample(self, num_samples, previous_state=None, evidence=None, evidence_mask=None, x=None):
        """Return num_samples samples from the proposa distribution, potentially conditioned on a previous state.
        Previous state is a tensor compatible with sampling, i.e. [num_relational_examples, num_samples, dim_state],
        where num_relational_examples is 1 in all task with a unique KB."""
        raise NotImplementedError("ProposalDistribution.sample() is an abstract method")

    def fit(self, y, **kwargs):
        """Run 1 GD step on the proposal distribution model given supervision y"""
        raise NotImplementedError("ProposalDistribution.fit() is an abstract method")

class GPUGibbsSampler(Sampler):

    def __init__(self, potential, num_variables, inter_sample_burn=1, num_chains = 10, initial_state=None, evidence = None, evidence_mask = None, flips=None, num_examples=1):

        self.potential = potential
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        self.num_examples = num_examples
        self.num_chains = num_chains

        if initial_state is None:
            self.current_state = tf.cast(tf.random.uniform(shape=[self.num_examples, self.num_chains, num_variables], minval=0,
                                                                           maxval=2, dtype=tf.int32), tf.float32)
        else:
            self.current_state = tf.cast(initial_state, tf.float32)
        self.evidence = evidence
        if evidence is not None:
            self.evidence = tf.tile(tf.expand_dims(evidence, axis=1), [1, num_chains, 1])
            self.evidence_mask =  tf.tile(tf.expand_dims(evidence_mask, axis=1), [1, num_chains, 1])
        self.flips = flips if flips is not None and flips > 0 else num_variables

    @tf.function
    def __sample(self, current_state, conditional_data=None, num_samples=None, minibatch = None):

        # Gibbs sampling in random scan mode
        # todo(giuseppe) allow ordered scan or other euristics from outside

        n_ex = self.num_examples if minibatch is None else len(minibatch)

        #todo(giuseppe) improve the speed of this shuffling and of the third dimension. It has slowed too much.
        # scan = tf.reshape(tf.range(ontology.num_variables, dtype=tf.int32), [-1, 1, 1])
        # scan = tf.tile(scan, [1, n_ex, ontology.num_chains]) #num_variables, num_examples, num_chains (needed for shuffling along the axis=0
        # K = tf.random.shuffle(scan)[:ontology.flips,:,:]
        # K = tf.transpose(K, [1, 2, 0])
        r = tf.random.shuffle(tf.range(self.num_variables, dtype=tf.int32))[:self.flips]

        for i in r:
            mask = tf.one_hot(i, depth=self.num_variables)
            mask = tf.reshape(mask, [1,1,-1])
            off_state = current_state * (1 - mask)
            on_state = off_state + mask
            rand = tf.random.uniform(shape=[n_ex, self.num_chains])

            potential_on = self.potential(on_state, conditional_data)
            potential_off = self.potential(off_state, conditional_data)


            p = tf.sigmoid(potential_on - potential_off)
            cond = tf.reshape(rand < p, shape=[n_ex, self.num_chains, 1])
            current_state = tf.where(cond, on_state, off_state)
            if self.evidence is not None:
                current_state = tf.where(self.evidence_mask, self.evidence, current_state)



        return current_state



    def sample(self, conditional_data=None, num_samples=None, minibatch=None):

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        for _ in range(self.inter_sample_burn):

            sample = self.__sample(current_state,conditional_data, num_samples, minibatch)
            if minibatch is None:
                self.current_state = sample
            else:
                self.current_state = tf.tensor_scatter_nd_update(self.current_state, tf.reshape(minibatch,[-1,1]), sample)

        return sample

class GPUGibbsSamplerBool(Sampler):

    def __init__(self, potential, num_variables, inter_sample_burn=1, num_chains = 10, initial_state=None, evidence = None, evidence_mask = None, flips=None, num_examples=1):

        self.potential = potential
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        self.num_examples = num_examples
        self.num_chains = num_chains

        if initial_state is None:
            self.current_state = tf.cast(tf.random.uniform(shape=[self.num_examples, self.num_chains, num_variables], minval=0,
                                                                           maxval=2, dtype=tf.int32), tf.bool)
        else:
            self.current_state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, self.num_chains, 1])
        self.evidence = evidence
        if evidence is not None:
            self.evidence = tf.tile(tf.expand_dims(evidence, axis=1), [1, num_chains, 1])
            self.evidence_mask =  tf.tile(tf.expand_dims(evidence_mask, axis=1), [1, num_chains, 1])
        self.flips = flips if flips is not None and flips > 0 else num_variables

    @tf.function
    def __sample(self, current_state, conditional_data=None, num_samples=None, minibatch = None):

        # Gibbs sampling in random scan mode
        # todo(giuseppe) allow ordered scan or other euristics from outside

        n_ex = self.num_examples if minibatch is None else len(minibatch)

        r = tf.random.shuffle(tf.range(self.num_variables, dtype=tf.int32))[:self.flips]

        for i in r:
            mask = tf.cast(tf.one_hot(i, depth=self.num_variables),dtype=tf.bool)
            mask = tf.reshape(mask, [1,1,-1])
            off_state = tf.logical_and(current_state, tf.logical_not(mask))
            on_state = tf.logical_or(off_state,mask)
            rand = tf.random.uniform(shape=[n_ex, self.num_chains])


            on_state_fl = tf.cast(on_state, tf.float32)
            off_state_fl = tf.cast(off_state, tf.float32)
            potential_on = self.potential(on_state_fl, conditional_data)
            potential_off = self.potential(off_state_fl, conditional_data)

            p = tf.sigmoid(potential_on - potential_off)
            cond = tf.reshape(rand < p, shape=[n_ex, self.num_chains, 1])
            current_state = tf.where(cond, on_state, off_state)
            if self.evidence is not None:
                current_state = tf.where(self.evidence_mask, self.evidence, current_state)



        return current_state



    def sample(self, conditional_data=None, num_samples=None, minibatch=None):

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        for _ in range(self.inter_sample_burn):

            sample = self.__sample(current_state,conditional_data, num_samples, minibatch)
            if minibatch is None:
                self.current_state = sample
            else:
                self.current_state = tf.tensor_scatter_nd_update(self.current_state, tf.reshape(minibatch,[-1,1]), sample)

        return tf.cast(sample, tf.float32)

class FactorizedGPUGibbsSamplerBool(Sampler):

    def __init__(self, potential, num_variables, factorization_idx, factorization_ids, inter_sample_burn=1, num_chains = 10, initial_state=None, evidence = None, evidence_mask = None, flips=None, num_examples=1):

        self.potential = potential
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        self.num_examples = num_examples
        self.num_chains = num_chains
        self.factorization_idx = tf.constant(factorization_idx)
        self.factorization_ids = tf.constant(factorization_ids)
        self.num_factorization_pairs = tf.constant(self.factorization_idx.shape[0])
        self.n_one_factors = tf.constant(self.factorization_idx.shape[1])
        self.num_variables_in_one_factor = tf.constant(self.factorization_idx.shape[2])

        if initial_state is None:
            # ontology.current_state = tf.cast(tf.random.uniform(shape=[ontology.num_examples, ontology.num_chains, num_variables], minval=0,
            #                                                                maxval=2, dtype=tf.int32), tf.bool)
            self.current_state = tf.zeros(shape=[self.num_examples, self.num_chains, num_variables],dtype= tf.bool)
        else:
            self.current_state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, self.num_chains, 1])

        self.evidence = evidence
        if evidence is not None:
            self.evidence = tf.expand_dims(evidence, axis=1)
            # ontology.evidence_mask =  tf.tile(tf.expand_dims(evidence_mask, axis=1), [1, num_chains, 1])
        self.flips = flips if flips is not None and flips > 0 else num_variables


    @tf.function
    def __sample(self, current_state, conditional_data=None, num_samples=None, minibatch = None):


        # TODO(giuseppe): see the link with minibacthes/conditional_data: here the factorization is fixed for the entire class

        n_ex = self.num_examples if minibatch is None else len(minibatch)

        id_0 = tf.random.uniform(shape=[self.flips, self.num_factorization_pairs], minval=0, maxval=self.n_one_factors,
                                 dtype=tf.int32)

        id_2 = tf.random.uniform(shape=[self.flips, self.num_factorization_pairs], minval=0,
                                 maxval=self.num_variables_in_one_factor, dtype=tf.int32)

        for i in range(self.flips):


            to_flip_idx = tf.gather(params=self.factorization_idx, indices=id_0[i], batch_dims=1)
            to_flip_idx = tf.gather(params=to_flip_idx, indices=id_2[i], batch_dims=1)
            to_flip_ids = tf.gather(params=self.factorization_ids, indices=id_0[i], batch_dims=1)


            masks = tf.cast(tf.one_hot(to_flip_idx, depth=self.num_variables),dtype=tf.bool)
            mask = tf.reduce_any(masks, axis=0)
            # mask = tf.tensor_scatter_nd_update(indices = tf.expand_dims(to_flip_idx,1), tensor = tf.zeros([ontology.num_variables],dtype=tf.bool), updates = tf.ones_like(to_flip_idx, dtype=tf.bool), )
            # mask = tf.cast(tf.one_hot(i, depth=ontology.num_variables),dtype=tf.bool)
            mask = tf.reshape(mask, [1,1,-1])
            off_state = tf.logical_and(current_state, tf.logical_not(mask))
            on_state = tf.logical_or(off_state,mask)
            rand = tf.random.uniform(shape=[n_ex, self.num_chains, self.num_factorization_pairs, 1])


            on_state_fl = tf.cast(on_state, tf.float32)
            off_state_fl = tf.cast(off_state, tf.float32)
            # ON STATE
            # on_state_fl_gr = ontology.potential.ground(on_state_fl, conditional_data)
            # potential_on = ontology.potential.call_on_groundings(on_state_fl_gr, conditional_data)
            # potential_on = tf.gather(potential_on, to_flip_ids, axis=-1)
            # potential_on = tf.reduce_sum(potential_on, axis=-1,keepdims=True)
            on_state_fl_gr = self.potential.ground(on_state_fl, conditional_data)

            on_state_fl_gr_filtered = osff = tf.gather(on_state_fl_gr, to_flip_ids, axis=-2)


            sh= osff.shape
            on_state_fl_gr_reshaped = tf.reshape(on_state_fl_gr_filtered, [sh[0], sh[1], sh[2]*sh[3], sh[-1]])

            potential_on = self.potential.call_on_groundings(on_state_fl_gr_reshaped, conditional_data)
            potential_on = tf.reshape(potential_on, [sh[0], sh[1], sh[2], sh[3]])
            potential_on = tf.reduce_sum(potential_on, axis=-1,keepdims=True)
            # print("Create on potential", time.time() - t0)



            # OFF STATE
            # off_state_fl_gr = ontology.potential.ground(off_state_fl, conditional_data)
            # potential_off = ontology.potential.call_on_groundings(off_state_fl_gr, conditional_data)
            # potential_off = tf.gather(potential_off, to_flip_ids, axis=-1)
            # potential_off = tf.reduce_sum(potential_off, axis=-1, keepdims=True)
            off_state_fl_gr = self.potential.ground(off_state_fl, conditional_data)
            off_state_fl_gr_filtered = osff = tf.gather(off_state_fl_gr, to_flip_ids, axis=-2)
            sh = osff.shape
            off_state_fl_gr_reshaped = tf.reshape(off_state_fl_gr_filtered, [sh[0], sh[1], sh[2] * sh[3], sh[-1]])
            potential_off = self.potential.call_on_groundings(off_state_fl_gr_reshaped, conditional_data)
            potential_off = tf.reshape(potential_off, [sh[0], sh[1], sh[2], sh[3]])
            potential_off = tf.reduce_sum(potential_off, axis=-1, keepdims=True)


            masks = tf.expand_dims(tf.expand_dims(masks,0),0)
            # off_state_multi = tf.tile(tf.expand_dims(off_state,-2), [1,1, ontology.num_factorization_pairs, 1])
            off_state_multi = tf.expand_dims(off_state,-2)
            on_state_multi = tf.logical_or(off_state_multi, masks)
            p = tf.sigmoid(potential_on - potential_off)
            cond = rand < p
            current_state_x = tf.where(cond, on_state_multi, off_state_multi)
            current_state_x = tf.reduce_any(current_state_x, axis=-2)
            current_state = current_state_x

            if self.evidence is not None:
                # current_state = tf.where(ontology.evidence_mask, ontology.evidence, current_state)
                current_state = tf.logical_or(self.evidence, current_state)



        return current_state


    def sample(self, conditional_data=None, num_samples=None, minibatch=None):

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        for _ in range(self.inter_sample_burn):

            sample = self.__sample(current_state,conditional_data, num_samples, minibatch)
            if minibatch is None:
                self.current_state = sample
            else:
                self.current_state = tf.tensor_scatter_nd_update(self.current_state, tf.reshape(minibatch,[-1,1]), sample)

        return tf.cast(sample, tf.float32)

class GPUGibbsSamplerV2(Sampler):


    def __init__(self, potential, num_variables, inter_sample_burn=1, num_chains = 10, initial_state=None, evidence = None, evidence_mask = None, flips=None, num_examples=1, coldness=1., parameters = None):

        self.potential = potential
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        self.num_examples = num_examples
        self.num_chains = num_chains
        self.coldness = coldness
        self.parameters = parameters

        if initial_state is None:
            self.current_state = tf.cast(tf.random.uniform(shape=[self.num_examples, self.num_chains, num_variables], minval=0,
                                                                           maxval=2, dtype=tf.int32), tf.float32)
        else:
            self.current_state = tf.tile(tf.expand_dims(tf.cast(initial_state, tf.float32), axis=1), [1, self.num_chains, 1])
        self.evidence = evidence
        if evidence is not None:
            self.evidence = evidence
            self.evidence_mask = tf.cast(evidence_mask, tf.int32)
        if flips is not None and flips > 0:
            self.flips = flips
        elif self.evidence is not None:
            self.flips = tf.reduce_max(tf.reduce_sum(1 - evidence_mask, axis=-1)) #check the maximum number of non-observed
        else:
            self.flips = num_variables


    @tf.function
    def __sample(self, current_state, conditional_data=None, num_samples=None, minibatch = None):


        # Gibbs sampling in random scan mode
        # todo(giuseppe) allow ordered scan or other euristics from outside

        n_ex = self.num_examples if minibatch is None else len(minibatch)

        # logits = tf.math.log((1 - ontology.evidence_mask) / tf.reduce_sum(1 - ontology.evidence_mask, axis=-1, keepdims=True))
        # samples = tf.random.categorical(logits=logits, num_samples=tf.cast(ontology.flips*ontology.num_chains, tf.int32), dtype=None, seed=None, name=None)
        # samples = tf.reshape(samples, [-1, ontology.num_chains, ontology.flips])
        # masks = tf.one_hot(samples, depth=ontology.num_variables)

        masks = tf.one_hot(tf.where(self.evidence_mask < 1)[:, 1], depth=self.num_variables)
        masks = tf.random.shuffle(masks)
        masks = tf.reshape(masks, [1,1,-1,self.num_variables])


        for i in range(self.flips):
            mask = masks[:,:,i,:]
            off_state = current_state * (1 - mask)
            on_state = off_state + mask

            rand = tf.random.uniform(shape=[n_ex, self.num_chains])

            potential_on = self.potential(on_state, conditional_data)
            potential_off = self.potential(off_state, conditional_data)

            p = tf.sigmoid(self.coldness*(potential_on - potential_off))
            cond = tf.reshape(rand < p, shape=[n_ex, self.num_chains, 1])
            current_state = tf.where(cond, on_state, off_state)


        return current_state



    def sample(self, conditional_data=None, num_samples=None, minibatch=None):

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        for _ in range(self.inter_sample_burn):

            sample = self.__sample(current_state, conditional_data, num_samples, minibatch)
            if minibatch is None:
                self.current_state = sample
            else:
                self.current_state = tf.tensor_scatter_nd_update(self.current_state, tf.reshape(minibatch,[-1,1]), sample)

        return sample

class GPUGibbsSamplerBoolVar(Sampler):

    def __init__(self, potential, num_variables, inter_sample_burn=1, num_chains = 10, initial_state=None, evidence = None, evidence_mask = None, flips=None, num_examples=1):

        self.potential = potential
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        self.num_examples = num_examples
        self.num_chains = num_chains

        if initial_state is None:
            self.current_state = tf.Variable(initial_value=tf.cast(tf.random.uniform(shape=[self.num_examples, self.num_chains, num_variables], minval=0,
                                                                           maxval=2, dtype=tf.int32), tf.bool))
        else:
            self.current_state = tf.Variable(initial_value=tf.cast(initial_state, tf.bool))
        self.evidence = evidence
        if evidence is not None:
            self.evidence = tf.tile(tf.expand_dims(evidence, axis=1), [1, num_chains, 1])
            self.evidence_mask =  tf.tile(tf.expand_dims(evidence_mask, axis=1), [1, num_chains, 1])
        self.flips = flips if flips is not None and flips > 0 else num_variables


    # @tf.function(experimental_compile=True)
    def sample(self, conditional_data=None, num_samples=None, minibatch=None):

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        for _ in range(self.inter_sample_burn):

            n_ex = self.num_examples if minibatch is None else minibatch.shape[0]

            r = tf.random.shuffle(tf.range(self.num_variables, dtype=tf.int32))[:self.flips]

            for i in range(self.flips):
                i = r[i]
                mask = tf.cast(tf.one_hot(i, depth=self.num_variables), dtype=tf.bool)
                mask = tf.reshape(mask, [1, 1, -1])
                off_state = tf.logical_and(current_state, tf.logical_not(mask))
                on_state = tf.logical_or(off_state, mask)
                rand = tf.random.uniform(shape=[n_ex, self.num_chains])

                on_state_fl = tf.cast(on_state, tf.float32)
                off_state_fl = tf.cast(off_state, tf.float32)
                potential_on = self.potential(on_state_fl, conditional_data)
                potential_off = self.potential(off_state_fl, conditional_data)

                p = tf.sigmoid(potential_on - potential_off)
                cond = tf.reshape(rand < p, shape=[n_ex, self.num_chains, 1])
                current_state = tf.where(cond, on_state, off_state)
                if self.evidence is not None:
                    current_state = tf.where(self.evidence_mask, self.evidence, current_state)

            if minibatch is None:
                self.current_state.assign(current_state)
            else:
                # ontology.current_state.scatter_nd_update(tf.reshape(minibatch,[-1,1]), current_state)
                self.current_state.assign(tf.tensor_scatter_nd_update(self.current_state, tf.reshape(minibatch,[-1,1]), current_state)
)

        return tf.cast(current_state, tf.float32)

class GPUGibbsSamplerWithTransformations(Sampler):


    def __init__(self, potential, transformations, num_variables, inter_sample_burn=1, num_chains = 10, initial_state=None, evidence = None, evidence_mask = None, flips=None, num_examples=1, coldness=1., parameters = None):

        self.potential = potential
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        self.num_examples = num_examples
        self.num_chains = num_chains
        self.coldness = coldness
        self.parameters = parameters
        self.transformations = transformations

        if initial_state is None:
            self.current_state = tf.zeros(shape=[self.num_examples, self.num_chains, num_variables], dtype= tf.float32)
            # ontology.current_state = tf.cast(tf.random.uniform(shape=[ontology.num_examples, ontology.num_chains, num_variables], minval=0,
            #                                                                maxval=2, dtype=tf.int32), tf.float32)
        else:
            self.current_state = tf.tile(tf.expand_dims(tf.cast(initial_state, tf.float32), axis=1), [1, self.num_chains, 1])
        self.evidence = evidence
        if evidence is not None:
            self.evidence = evidence
            self.evidence_mask = tf.cast(evidence_mask, tf.int32)
        if flips is not None and flips > 0:
            self.flips = flips
        elif self.evidence is not None:
            self.flips = tf.reduce_max(tf.reduce_sum(1 - evidence_mask, axis=-1)) #check the maximum number of non-observed
        else:
            self.flips = num_variables


        self.states = []

    @tf.function
    def _inner_(self, y,  m, alsoNone):
        m2 = tf.reduce_sum(m, -2)
        M = tf.expand_dims(tf.expand_dims(m, 0), 0)
        t = t1 = tf.where(m2 > 0, tf.zeros_like(y), y)
        t = tf.expand_dims(t, -2)
        t = tf.where(M > 0, tf.ones_like(t), t)
        if alsoNone:
            t1 = tf.expand_dims(t1, -2)
            t = tf.concat((t1, t), axis=-2)
        return t



    @tf.function
    def __sample(self, current_state, conditional_data=None, num_samples=None, minibatch = None):

        # for i in range(ontology.flips):
        for m, alsoNone in np.random.permutation(self.transformations):
            t = self._inner_(current_state, m,alsoNone)
            potentials = temp = self.potential(t, conditional_data)
            logits = tf.reshape(potentials, [-1, potentials.shape[-1]])
            p = tf.nn.softmax(logits)
            logits = tf.math.log(p)
            sample = tf.random.categorical(logits=logits, num_samples=1, dtype=None, seed=None, name=None)
            shape = temp.shape[:-1] + [1]
            sample = tf.reshape(sample, shape)
            current_state = tf.gather(params=t, indices=sample, batch_dims=2)
            current_state = tf.squeeze(current_state, axis=-2)
        return current_state



    def sample(self, conditional_data=None, num_samples=None, minibatch=None):

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        for _ in range(self.inter_sample_burn):

            sample = self.__sample(current_state, conditional_data, num_samples, minibatch)
            if minibatch is None:
                self.current_state = sample
            else:
                self.current_state = tf.tensor_scatter_nd_update(self.current_state, tf.reshape(minibatch,[-1,1]), sample)

        return sample
