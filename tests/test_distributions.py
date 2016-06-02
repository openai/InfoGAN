from __future__ import print_function
from __future__ import absolute_import

from nose2.tools import such
from misc.distributions import Categorical, Gaussian, Product, Bernoulli
import numpy as np
import tensorflow as tf

sess = tf.Session()


def random_softmax(ndim):
    x = np.random.uniform(size=(ndim,))
    x = x - np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    return np.cast['float32'](x)


with such.A("Product Distribution") as it:
    dist1 = Product([Categorical(5), Categorical(3)])
    dist2 = Product([Gaussian(5), dist1])


    @it.should
    def test_dist_info_keys():
        it.assertEqual(set(dist1.dist_info_keys), {"id_0_prob", "id_1_prob"})
        it.assertEqual(set(dist2.dist_info_keys), {"id_0_mean", "id_0_stddev",
                                                   "id_1_id_0_prob", "id_1_id_1_prob"})


    @it.should
    def test_kl_sym():
        old_id_0_prob = np.array([random_softmax(5)])
        old_id_1_prob = np.array([random_softmax(3)])
        new_id_0_prob = np.array([random_softmax(5)])
        new_id_1_prob = np.array([random_softmax(3)])
        old_dist_info_vars = dict(
            id_0_prob=tf.constant(old_id_0_prob),
            id_1_prob=tf.constant(old_id_1_prob)
        )
        new_dist_info_vars = dict(
            id_0_prob=tf.constant(new_id_0_prob),
            id_1_prob=tf.constant(new_id_1_prob)
        )
        np.testing.assert_allclose(
            dist1.kl(old_dist_info_vars, new_dist_info_vars).eval(session=sess),
            Categorical(5).kl(dict(prob=old_id_0_prob), dict(prob=new_id_0_prob)).eval(session=sess) +
            Categorical(3).kl(dict(prob=old_id_1_prob), dict(prob=new_id_1_prob)).eval(session=sess)
        )

it.createTests(globals())

with such.A("Categorical") as it:
    @it.should
    def test_categorical():
        cat = Categorical(3)
        new_prob = np.array(
            [random_softmax(3), random_softmax(3)],
        )
        old_prob = np.array(
            [random_softmax(3), random_softmax(3)],
        )

        x = np.array([
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        new_prob_sym = tf.constant(new_prob)
        old_prob_sym = tf.constant(old_prob)

        x_sym = tf.constant(x)

        new_info_sym = dict(prob=new_prob_sym)
        old_info_sym = dict(prob=old_prob_sym)

        np.testing.assert_allclose(
            cat.kl(new_info_sym, new_info_sym).eval(session=sess),
            np.array([0., 0.])
        )
        np.testing.assert_allclose(
            cat.kl(old_info_sym, new_info_sym).eval(session=sess),
            np.sum(old_prob * (np.log(old_prob + 1e-8) - np.log(new_prob + 1e-8)), axis=-1)
        )
        np.testing.assert_allclose(
            cat.logli(x_sym, old_info_sym).eval(session=sess),
            [np.log(old_prob[0][1] + 1e-8), np.log(old_prob[1][2] + 1e-8)],
            rtol=1e-5
        )

it.createTests(globals())

with such.A("Bernoulli") as it:
    @it.should
    def test_bernoulli():
        bernoulli = Bernoulli(3)

        new_p = np.array([[0.5, 0.5, 0.5], [.9, .9, .9]], dtype=np.float32)
        old_p = np.array([[.9, .9, .9], [.1, .1, .1]], dtype=np.float32)

        x = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.float32)

        x_sym = tf.constant(x)
        new_p_sym = tf.constant(new_p)
        old_p_sym = tf.constant(old_p)

        new_info = dict(p=new_p)
        old_info = dict(p=old_p)

        new_info_sym = dict(p=new_p_sym)
        old_info_sym = dict(p=old_p_sym)

        # np.testing.assert_allclose(
        #     np.sum(bernoulli.entropy(dist_info=new_info)),
        #     np.sum(- new_p * np.log(new_p + 1e-8) - (1 - new_p) * np.log(1 - new_p + 1e-8)),
        # )

        # np.testing.assert_allclose(
        #     np.sum(bernoulli.kl(old_info_sym, new_info_sym).eval()),
        #     np.sum(old_p * (np.log(old_p + 1e-8) - np.log(new_p + 1e-8)) + (1 - old_p) * (np.log(1 - old_p + 1e-8) -
        #                                                                                   np.log(1 - new_p + 1e-8))),
        # )
        # np.testing.assert_allclose(
        #     np.sum(bernoulli.kl(old_info, new_info)),
        #     np.sum(old_p * (np.log(old_p + 1e-8) - np.log(new_p + 1e-8)) + (1 - old_p) * (np.log(1 - old_p + 1e-8) -
        #                                                                                   np.log(1 - new_p + 1e-8))),
        # )
        # np.testing.assert_allclose(
        #     bernoulli.likelihood_ratio_sym(x_sym, old_info_sym, new_info_sym).eval(),
        #     np.prod((x * new_p + (1 - x) * (1 - new_p)) / (x * old_p + (1 - x) * (1 - old_p) + 1e-8), axis=-1)
        # )
        np.testing.assert_allclose(
            bernoulli.logli(x_sym, old_info_sym).eval(session=sess),
            np.sum(x * np.log(old_p + 1e-8) + (1 - x) * np.log(1 - old_p + 1e-8), axis=-1)
        )
        # np.testing.assert_allclose(
        #     bernoulli.log_likelihood(x, old_info),
        #     np.sum(x * np.log(old_p + 1e-8) + (1 - x) * np.log(1 - old_p + 1e-8), axis=-1)
        # )

it.createTests(globals())
