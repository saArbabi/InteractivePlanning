from tensorflow.keras.layers import Dense, LSTM, Concatenate, TimeDistributed
from keras import backend as K
from importlib import reload
from models.core import abstract_model
reload(abstract_model)
from models.core.abstract_model import  AbstractModel
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class CAE(AbstractModel):
    def __init__(self, config, model_use):
        super(CAE, self).__init__(config)
        self.pred_step_n = config['data_config']['pred_step_n']
        self.enc_model = HistoryEncoder()
        self.dec_model = FutureDecoder(config, model_use)
        self.dec_model.total_batch_count = None

    def callback_def(self):
        self.train_llloss_m = tf.keras.metrics.Mean()
        self.train_llloss_y = tf.keras.metrics.Mean()
        self.train_llloss_f = tf.keras.metrics.Mean()
        self.train_llloss_fadj = tf.keras.metrics.Mean()

        self.test_llloss_m = tf.keras.metrics.Mean()
        self.test_llloss_y = tf.keras.metrics.Mean()
        self.test_llloss_f = tf.keras.metrics.Mean()
        self.test_llloss_fadj = tf.keras.metrics.Mean()

    def train_loop(self, data_objs):
        # for seq_len in range(2, 3):
        for seq_len in range(1, self.pred_step_n+1):
            train_seq_data = [data_objs[0][seq_len],
                                data_objs[1][seq_len][0],
                                data_objs[1][seq_len][1],
                                data_objs[1][seq_len][2],
                                data_objs[1][seq_len][3],
                                data_objs[2][seq_len][0],
                                data_objs[2][seq_len][1],
                                data_objs[2][seq_len][2],
                                data_objs[2][seq_len][3]]

            train_ds = self.batch_data(train_seq_data)
            for s, targ_m, targ_y, targ_f, targ_fadj, cond_m, cond_y, cond_f, cond_fadj in train_ds:
                self.train_step(s, [targ_m, targ_y, targ_f, targ_fadj], \
                                [cond_m, cond_y, cond_f, cond_fadj])

    def test_loop(self, data_objs):
        # for seq_len in range(2, 3):
        for seq_len in range(1, self.pred_step_n+1):
            test_seq_data = [data_objs[0][seq_len],
                                data_objs[1][seq_len][0],
                                data_objs[1][seq_len][1],
                                data_objs[1][seq_len][2],
                                data_objs[1][seq_len][3],
                                data_objs[2][seq_len][0],
                                data_objs[2][seq_len][1],
                                data_objs[2][seq_len][2],
                                data_objs[2][seq_len][3]]

            test_ds = self.batch_data(test_seq_data)
            for s, targ_m, targ_y, targ_f, targ_fadj, cond_m, cond_y, cond_f, cond_fadj in test_ds:
                self.test_step(s, [targ_m, targ_y, targ_f, targ_fadj], \
                                [cond_m, cond_y, cond_f, cond_fadj])

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, states, targets, conditions):
        with tf.GradientTape() as tape:
            gmm_m, gmm_y, gmm_f, gmm_fadj = self([states, conditions])
            llloss_m = self.log_loss(targets[0], gmm_m)
            llloss_y = self.log_loss(targets[1], gmm_y)
            llloss_f = self.log_loss(targets[2], gmm_f)
            llloss_fadj = self.log_loss(targets[3], gmm_fadj)
            loss = sum([llloss_m, llloss_y, llloss_f, llloss_fadj])

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_llloss_m.reset_states()
        self.train_llloss_y.reset_states()
        self.train_llloss_f.reset_states()
        self.train_llloss_fadj.reset_states()
        self.train_llloss_m(llloss_m)
        self.train_llloss_y(llloss_y)
        self.train_llloss_f(llloss_f)
        self.train_llloss_fadj(llloss_fadj)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, states, targets, conditions):
        gmm_m, gmm_y, gmm_f, gmm_fadj = self([states, conditions])
        llloss_m = self.log_loss(targets[0], gmm_m)
        llloss_y = self.log_loss(targets[1], gmm_y)
        llloss_f = self.log_loss(targets[2], gmm_f)
        llloss_fadj = self.log_loss(targets[3], gmm_fadj)
        loss = sum([llloss_m, llloss_y, llloss_f, llloss_fadj])

        self.test_llloss_m.reset_states()
        self.test_llloss_y.reset_states()
        self.test_llloss_f.reset_states()
        self.test_llloss_fadj.reset_states()
        self.test_llloss_m(llloss_m)
        self.test_llloss_y(llloss_y)
        self.test_llloss_f(llloss_f)
        self.test_llloss_fadj(llloss_fadj)

    def call(self, inputs):
        # Defines the computation from inputs to outputs
        # input[0] = state obs
        # input[1] = conditions
        encoder_states = self.enc_model(inputs[0])
        return self.dec_model([inputs[1], encoder_states])

class HistoryEncoder(tf.keras.Model):
    def __init__(self):
        super(HistoryEncoder, self).__init__(name="HistoryEncoder")
        self.enc_units = 128
        self.architecture_def()

    def architecture_def(self):
        self.lstm_layer_1 = LSTM(self.enc_units, return_sequences=True)
        self.lstm_layer_2 = LSTM(self.enc_units, return_sequences=True, return_state=True)

    def call(self, inputs):
        whole_seq_output = self.lstm_layer_1(inputs)
        _, state_h, state_c = self.lstm_layer_2(whole_seq_output)
        return [state_h, state_c]


class FutureDecoder(tf.keras.Model):
    def __init__(self, config, model_use):
        super(FutureDecoder, self).__init__(name="FutureDecoder")
        self.dec_units = 128
        self.components_n = config['model_config']['components_n'] # number of Mixtures
        self.model_use = model_use # can be training or inference

        if 'allowed_error' in config['model_config']:
            self.allowed_error = config['model_config']['allowed_error']
        else:
            self.allowed_error = 0
        self.architecture_def()

    def architecture_def(self):
        self.pvector = Concatenate(name="output") # parameter vector
        self.lstm_layer_m = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_y = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_f = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.lstm_layer_fadj = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.gmm_linear_m = TimeDistributed(Dense(self.components_n*6))
        self.gmm_linear_y = TimeDistributed(Dense(self.components_n*6))
        self.gmm_linear_f = TimeDistributed(Dense(self.components_n*6))
        self.gmm_linear_fadj = TimeDistributed(Dense(self.components_n*6))

        """Merger vehicle
        """
        self.alphas_m = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.rhos_m = Dense(self.components_n, activation=K.tanh, name="rhos")
        self.mus_lon_m = Dense(self.components_n, name="mus_long")
        self.sigmas_lon_m = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_m = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_m = Dense(self.components_n, activation=K.exp, name="sigmas_lat")

        """Yielder vehicle
        """
        self.alphas_y = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.rhos_y = Dense(self.components_n, activation=K.tanh, name="rhos")
        self.mus_lon_y = Dense(self.components_n, name="mus_long")
        self.sigmas_lon_y = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_y = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_y = Dense(self.components_n, activation=K.exp, name="sigmas_lat")

        """F vehicle
        """
        self.alphas_f = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.rhos_f = Dense(self.components_n, activation=K.tanh, name="rhos")
        self.mus_lon_f = Dense(self.components_n, name="mus_long")
        self.sigmas_lon_f = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_f = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_f = Dense(self.components_n, activation=K.exp, name="sigmas_lat")

        """Fadj vehicle
        """
        self.alphas_fadj = Dense(self.components_n, activation=K.softmax, name="alphas")
        self.rhos_fadj = Dense(self.components_n, activation=K.tanh, name="rhos")
        self.mus_lon_fadj = Dense(self.components_n, name="mus_long")
        self.sigmas_lon_fadj = Dense(self.components_n, activation=K.exp, name="sigmas_long")
        self.mus_lat_fadj = Dense(self.components_n, name="mus_lat")
        self.sigmas_lat_fadj = Dense(self.components_n, activation=K.exp, name="sigmas_lat")

    def concat_vecs(self, step_gauss_param_vec, veh_gauss_param_vec, step):
        """Use for concatinating gmm parameters across time-steps
        """
        if step == 0:
            return step_gauss_param_vec
        else:
            veh_gauss_param_vec = tf.concat([veh_gauss_param_vec, step_gauss_param_vec], axis=1)
            return veh_gauss_param_vec

    def axis2_conc(self, items_list):
        return tf.concat(items_list, axis=-1)

    def teacher_check(self, true_action, sampled_action):
        error = tf.math.abs(tf.math.subtract(sampled_action, true_action))
        less = tf.cast(tf.math.less(error, self.allowed_error), dtype='float')
        greater = tf.cast(tf.math.greater_equal(error, self.allowed_error), dtype='float')
        return  tf.math.add(tf.multiply(greater, true_action), tf.multiply(less, sampled_action))

    def sample_action_lik_____(self, gmm):
        action = gmm.sample(1)
        likelihood = gmm.prob(action)
        return tf.reshape(action, [batch_size, 1, 1]), tf.reshape(likelihood, [batch_size, 1, 1])

    def sample_action(self, gmm):
        return tf.clip_by_value(tf.squeeze(gmm.sample(1), axis=0), -5, 5)

    def get_pdf(self, gmm_params):
        # tf.print(tf.shape(gmm_params))
        # tf.print(tf.reduce_min(tf.abs(gmm_params)))
        # tf.debugging.check_numerics(gmm_params, message='Checking gmm_params')
        alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat = \
                                                tf.split(gmm_params, 6, axis=2)

        covar = sigmas_lon*sigmas_lat*rhos
        col1 = tf.stack([sigmas_lon**2, covar], axis=3, name='stack')
        col2 = tf.stack([covar, sigmas_lat**2], axis=3, name='stack')
        cov = tf.stack([col1, col2], axis=3, name='cov')
        mus = tf.stack([mus_lon, mus_lat], axis=3, name='mus')

        mvn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alphas),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=mus,
                covariance_matrix=cov))
        return mvn

    def call(self, inputs):
        # input[0] = conditions, shape = (batch, steps_n, feature_size)
        # input[1] = encoder states
        cond_m, cond_y, cond_f, cond_fadj = inputs[0]
        state_h, state_c = inputs[1] # encoder cell state

        if self.model_use == 'training':
            batch_size = tf.shape(cond_m)[0] # dynamiclaly assigned
            steps_n = tf.shape(cond_m)[1] # dynamiclaly assigned

        elif self.model_use == 'inference':
            batch_size = tf.constant(self.traj_n)
            steps_n = tf.constant(self.steps_n)

        enc_h = tf.reshape(state_h, [batch_size, 1, self.dec_units]) # encoder hidden state
        state_h_m = state_h
        state_c_m = state_c
        state_h_y = state_h
        state_c_y = state_c
        state_h_f = state_h
        state_c_f = state_c
        state_h_fadj = state_h
        state_c_fadj = state_c

        # Initialize param vector
        gauss_params_seq_m = tf.zeros([0, 0, 0], dtype=tf.float32)
        gauss_params_seq_y = tf.zeros([0, 0, 0], dtype=tf.float32)
        gauss_params_seq_f = tf.zeros([0, 0, 0], dtype=tf.float32)
        gauss_params_seq_fadj = tf.zeros([0, 0, 0], dtype=tf.float32)

        act_m = cond_m[:, 0:1, :]
        act_y = cond_y[:, 0:1, :]
        act_f = cond_f[:, 0:1, :]
        act_fadj = cond_fadj[:, 0:1, :]

        if self.model_use == 'training':
            for step in range(steps_n):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                    (gauss_params_seq_m, tf.TensorShape([None,None,None])),
                    (gauss_params_seq_y, tf.TensorShape([None,None,None])),
                    (gauss_params_seq_f, tf.TensorShape([None,None,None])),
                    (gauss_params_seq_fadj, tf.TensorShape([None,None,None]))])

                if step > 0:
                    true_act_m = cond_m[:, step:step+1, :]
                    true_act_y = cond_y[:, step:step+1, :]
                    true_act_f = cond_f[:, step:step+1, :]
                    true_act_fadj = cond_fadj[:, step:step+1, :]
                    act_m = self.teacher_check(true_act_m, act_m)
                    act_y = self.teacher_check(true_act_y, act_y)
                    act_f = self.teacher_check(true_act_f, act_f)
                    act_fadj = self.teacher_check(true_act_fadj, act_fadj)

                step_cond_m = self.axis2_conc([act_m,
                                                    act_y,
                                                    act_f,
                                                    act_fadj])

                step_cond_y = self.axis2_conc([act_m,
                                                act_y,
                                                act_fadj])

                step_cond_f = act_f
                step_cond_fadj = act_fadj


                """Merger vehicle
                """
                outputs, state_h_m, state_c_m = self.lstm_layer_m(\
                                              self.axis2_conc([step_cond_m, enc_h]),
                                              initial_state=[state_h_m, state_c_m])
                outputs = self.gmm_linear_m(outputs)
                alphas = self.alphas_m(outputs)
                rhos = self.rhos_m(outputs)
                mus_lon = self.mus_lon_m(outputs)
                sigmas_lon = self.sigmas_lon_m(outputs)
                mus_lat = self.mus_lat_m(outputs)
                sigmas_lat = self.sigmas_lat_m(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_m = self.pvector(gmm_params)
                act_m = self.sample_action(self.get_pdf(gauss_params_m))

                """Yielder vehicle
                """
                outputs, state_h_y, state_c_y = self.lstm_layer_y(\
                                              self.axis2_conc([step_cond_y, enc_h]),
                                              initial_state=[state_h_y, state_c_y])
                outputs = self.gmm_linear_y(outputs)
                alphas = self.alphas_y(outputs)
                rhos = self.rhos_y(outputs)
                mus_lon = self.mus_lon_y(outputs)
                sigmas_lon = self.sigmas_lon_y(outputs)
                mus_lat = self.mus_lat_y(outputs)
                sigmas_lat = self.sigmas_lat_y(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_y = self.pvector(gmm_params)
                act_y = self.sample_action(self.get_pdf(gauss_params_y))
                """F vehicle
                """
                outputs, state_h_f, state_c_f = self.lstm_layer_f(\
                                              self.axis2_conc([step_cond_f, enc_h]),
                                              initial_state=[state_h_f, state_c_f])
                outputs = self.gmm_linear_f(outputs)
                alphas = self.alphas_f(outputs)
                rhos = self.rhos_f(outputs)
                mus_lon = self.mus_lon_f(outputs)
                sigmas_lon = self.sigmas_lon_f(outputs)
                mus_lat = self.mus_lat_f(outputs)
                sigmas_lat = self.sigmas_lat_f(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_f = self.pvector(gmm_params)
                act_f = self.sample_action(self.get_pdf(gauss_params_f))

                """Fadj vehicle
                """
                outputs, state_h_fadj, state_c_fadj = self.lstm_layer_fadj(\
                                              self.axis2_conc([step_cond_fadj, enc_h]),
                                              initial_state=[state_h_fadj, state_c_fadj])
                outputs = self.gmm_linear_fadj(outputs)
                alphas = self.alphas_fadj(outputs)
                rhos = self.rhos_fadj(outputs)
                mus_lon = self.mus_lon_fadj(outputs)
                sigmas_lon = self.sigmas_lon_fadj(outputs)
                mus_lat = self.mus_lat_fadj(outputs)
                sigmas_lat = self.sigmas_lat_fadj(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_fadj = self.pvector(gmm_params)
                act_fadj = self.sample_action(self.get_pdf(gauss_params_fadj))

                if step == 0:
                    gauss_params_seq_m = gauss_params_m
                    gauss_params_seq_y = gauss_params_y
                    gauss_params_seq_f = gauss_params_f
                    gauss_params_seq_fadj = gauss_params_fadj
                else:
                    gauss_params_seq_m = tf.concat([gauss_params_seq_m, gauss_params_m], axis=1)
                    gauss_params_seq_y = tf.concat([gauss_params_seq_y, gauss_params_y], axis=1)
                    gauss_params_seq_f = tf.concat([gauss_params_seq_f, gauss_params_f], axis=1)
                    gauss_params_seq_fadj = tf.concat([gauss_params_seq_fadj, gauss_params_fadj], axis=1)

            gmm_m = self.get_pdf(gauss_params_seq_m)
            gmm_y = self.get_pdf(gauss_params_seq_y)
            gmm_f = self.get_pdf(gauss_params_seq_f)
            gmm_fadj = self.get_pdf(gauss_params_seq_fadj)

            return gmm_m, gmm_y, gmm_f, gmm_fadj

        elif self.model_use == 'inference':
            for step in range(steps_n):
                tf.autograph.experimental.set_loop_options(shape_invariants=[
                    (gauss_params_seq_m, tf.TensorShape([None,None,None])),
                    (gauss_params_seq_y, tf.TensorShape([None,None,None])),
                    (gauss_params_seq_f, tf.TensorShape([None,None,None])),
                    (gauss_params_seq_fadj, tf.TensorShape([None,None,None]))])

                step_cond_m = self.axis2_conc([act_m,
                                                    act_y,
                                                    act_f,
                                                    act_fadj])

                step_cond_y = self.axis2_conc([act_m,
                                                act_y,
                                                act_fadj])

                step_cond_f = act_f
                step_cond_fadj = act_fadj

                """Merger vehicle
                """
                outputs, state_h_m, state_c_m = self.lstm_layer_m(\
                                              self.axis2_conc([step_cond_m, enc_h]),
                                              initial_state=[state_h_m, state_c_m])
                outputs = self.gmm_linear_m(outputs)
                alphas = self.alphas_m(outputs)
                rhos = self.rhos_m(outputs)
                mus_lon = self.mus_lon_m(outputs)
                sigmas_lon = self.sigmas_lon_m(outputs)
                mus_lat = self.mus_lat_m(outputs)
                sigmas_lat = self.sigmas_lat_m(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_m = self.pvector(gmm_params)
                act_m = self.sample_action(self.get_pdf(gauss_params_m))

                """Yielder vehicle
                """
                outputs, state_h_y, state_c_y = self.lstm_layer_y(\
                                              self.axis2_conc([step_cond_y, enc_h]),
                                              initial_state=[state_h_y, state_c_y])
                outputs = self.gmm_linear_y(outputs)
                alphas = self.alphas_y(outputs)
                rhos = self.rhos_y(outputs)
                mus_lon = self.mus_lon_y(outputs)
                sigmas_lon = self.sigmas_lon_y(outputs)
                mus_lat = self.mus_lat_y(outputs)
                sigmas_lat = self.sigmas_lat_y(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_y = self.pvector(gmm_params)
                act_y = self.sample_action(self.get_pdf(gauss_params_y))
                """F vehicle
                """
                outputs, state_h_f, state_c_f = self.lstm_layer_f(\
                                              self.axis2_conc([step_cond_f, enc_h]),
                                              initial_state=[state_h_f, state_c_f])
                outputs = self.gmm_linear_f(outputs)
                alphas = self.alphas_f(outputs)
                rhos = self.rhos_f(outputs)
                mus_lon = self.mus_lon_f(outputs)
                sigmas_lon = self.sigmas_lon_f(outputs)
                mus_lat = self.mus_lat_f(outputs)
                sigmas_lat = self.sigmas_lat_f(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_f = self.pvector(gmm_params)
                act_f = self.sample_action(self.get_pdf(gauss_params_f))
                """Fadj vehicle
                """
                outputs, state_h_fadj, state_c_fadj = self.lstm_layer_fadj(\
                                              self.axis2_conc([step_cond_fadj, enc_h]),
                                              initial_state=[state_h_fadj, state_c_fadj])
                outputs = self.gmm_linear_fadj(outputs)
                alphas = self.alphas_fadj(outputs)
                rhos = self.rhos_fadj(outputs)
                mus_lon = self.mus_lon_fadj(outputs)
                sigmas_lon = self.sigmas_lon_fadj(outputs)
                mus_lat = self.mus_lat_fadj(outputs)
                sigmas_lat = self.sigmas_lat_fadj(outputs)

                gmm_params = [alphas, rhos, mus_lon, sigmas_lon, mus_lat, sigmas_lat]
                gauss_params_fadj = self.pvector(gmm_params)
                act_fadj = self.sample_action(self.get_pdf(gauss_params_fadj))

                if step == 0:
                    gauss_params_seq_m = gauss_params_m
                    act_seq_m = act_m
                    act_seq_y = act_y
                    act_seq_f = act_f
                    act_seq_fadj = act_fadj
                else:
                    gauss_params_seq_m = tf.concat([gauss_params_seq_m, gauss_params_m], axis=1)
                    act_seq_m = tf.concat([act_seq_m, act_m], axis=1)
                    act_seq_y = tf.concat([act_seq_y, act_y], axis=1)
                    act_seq_f = tf.concat([act_seq_f, act_f], axis=1)
                    act_seq_fadj = tf.concat([act_seq_fadj, act_fadj], axis=1)

            actions = [act_seq_m, act_seq_y, act_seq_f, act_seq_fadj]
            gmm_m = self.get_pdf(gauss_params_seq_m)
            return actions, gmm_m
