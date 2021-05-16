import tensorflow as tf

class BooleanLogic():

    @staticmethod
    def cast(y):
        return tf.cast(y, tf.bool)

    @staticmethod
    def _not(args):
        assert len(args)==1, "N-Ary negation not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        return tf.logical_not(args[0])

    @staticmethod
    def _and(args):
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.stack(args, axis=-1)
        return tf.reduce_all(t, axis=-1)

    @staticmethod
    def _or(args):
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.stack(args, axis=-1)
        return tf.reduce_any(t, axis=-1)

    @staticmethod
    def _implies(args):
        assert len(args)==2, "N-Ary implies not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.logical_or(tf.logical_not(args[0]), args[1])
        return t

    @staticmethod
    def _iff(args):
        assert len(args) == 2, "N-Ary iff not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.equal(args[0], args[1])
        return t

    @staticmethod
    def _xor(args):
        assert len(args) == 2, "N-Ary xor not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.math.logical_xor(args[0], args[1])
        return t


class LukasiewiczLogic():

    @staticmethod
    def cast(y):
        return y


    @staticmethod
    def _not(args):
        assert len(args)==1, "N-Ary negation not defined"
        return 1 - args[0]

    @staticmethod
    def _and(args):
        t = tf.stack(args, axis=-1)
        return tf.reduce_sum(t - 1, axis=-1) + 1

    @staticmethod
    def _or(args):
        t = tf.stack(args, axis=-1)
        return tf.minimum(1., tf.reduce_sum(t, axis=-1))

    @staticmethod
    def _implies(args):
        assert len(args)==2, "N-Ary implies not defined"
        t = tf.minimum(1., 1 - args[0] + args[1])
        return t

    @staticmethod
    def _iff(args):
        t = 1 - tf.abs(args[0] - args[1])
        return t

    @staticmethod
    def _xor(args):
        assert len(args) == 2, "N-Ary xor not defined"
        return tf.abs(args[0] - args[1])


class ProductLogic():

    @staticmethod
    def cast(y):
        return y

    @staticmethod
    def _not(args):
        assert len(args)==1, "N-Ary negation not defined"
        return 1 - args[0]

    @staticmethod
    def _and(args):
        t = tf.stack(args, axis=-1)
        return tf.reduce_prod(t - 1, axis=-1) + 1

    @staticmethod
    def _or(args):
        assert len(args)==1, "N-Ary or not defined for product t-norm"
        return args[0] + args[1] + args[0]*args[1]

    @staticmethod
    def _implies(args):
        assert len(args)==2, "N-Ary implies not defined"
        a = args[0]
        b = args[1]
        return tf.where(a > b, b / (a + 1e-12), tf.ones_like(a))

    @staticmethod
    def _iff(args):
        assert len(args) == 2, "N-Ary <-> not defined"
        a = args[0]
        b = args[1]
        return 1 - a *(1-b) + (1-a)*b - a *(1-b)*(1-a)*b

    @staticmethod
    def _xor(args):
        assert len(args) == 2, "N-Ary xor not defined"
        a = args[0]
        b = args[1]
        return a *(1-b) + (1-a)*b - a *(1-b)*(1-a)*b