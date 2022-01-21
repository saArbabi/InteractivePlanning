from tensorflow.keras.layers import Dense, Concatenate
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class MLP(AbstractModel):
    def __init__(self, config=None):
        super(MLP, self).__init__(config)
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.architecture_def()

    def callback_def(self):
        self.train_llloss_m = tf.keras.metrics.Mean()
        self.test_llloss_m = tf.keras.metrics.Mean()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            gmm_m = self(states)
            loss = self.log_loss(targets, gmm_m)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_llloss_m.reset_states()
        self.train_llloss_m(loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets):
        gmm_m = self(states)
        loss = self.log_loss(targets, gmm_m)
        self.test_llloss_m.reset_states()
        self.test_llloss_m(loss)

    def log_loss(self, act_true, gmm_m):
        likelihood = gmm_m.log_prob(act_true)
        return -tf.reduce_mean(likelihood)

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.layer_1 = Dense(400, activation=K.relu)
        self.layer_2 = Dense(400, activation=K.relu)
        self.layer_3 = Dense(400, activation=K.relu)
        self.layer_4 = Dense(400, activation=K.relu)
        self.alphas_m = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.rhos_m = Dense(self.components_n, activation=K.tanh, name="rhos")
        self.mus_lon_m = Dense(self.components_n, name="mus_long")
        self.sigmas_lon_m = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_m = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_m = Dense(self.components_n, activation=K.exp, name="sigmas_lat")

    def train_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.train_step(s, t)

    def test_loop(self, data_objs):
        train_ds = self.batch_data(data_objs)
        for s, t in train_ds:
            self.test_step(s, t)

    def get_pdf(self, gmm_params):
        # tf.print(tf.shape(gmm_params))
        # tf.print(tf.reduce_min(tf.abs(gmm_params)))
        # tf.debugging.check_numerics(gmm_params, message='Checking gmm_params')
        alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat = \
                                                tf.split(gmm_params, 6, axis=1)

        covar = sigmas_lon*sigmas_lat*rhos
        col1 = tf.stack([sigmas_lon**2, covar], axis=2, name='stack')
        col2 = tf.stack([covar, sigmas_lat**2], axis=2, name='stack')
        cov = tf.stack([col1, col2], axis=2, name='cov')
        mus = tf.stack([mus_lon, mus_lat], axis=2, name='mus')

        mvn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alphas),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=mus,
                covariance_matrix=cov))
        return mvn

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        outputs = self.layer_4(x)

        alphas = self.alphas_m(outputs)
        rhos = self.rhos_m(outputs)
        mus_lon = self.mus_lon_m(outputs)
        sigmas_lon = self.sigmas_lon_m(outputs)
        mus_lat = self.mus_lat_m(outputs)
        sigmas_lat = self.sigmas_lat_m(outputs)

        gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
        gauss_params = self.pvector(gmm_params)
        gmm_m = self.get_pdf(gauss_params)
        return gmm_m
