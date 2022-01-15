import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class AbstractModel(tf.keras.Model):
    def __init__(self, config):
        super(AbstractModel, self).__init__(name="AbstractModel")
        self.callback_def()
        if config:
            self.config = config['model_config']
            self.optimizer = tf.optimizers.Adam(self.config['learning_rate'])
            self.batch_size = self.config['batch_size']

    def architecture_def(self):
        raise NotImplementedError()

    def callback_def(self):
        raise NotImplementedError()

    def batch_data(self, sets):
        data_slices = tuple([tf.cast(set, dtype='float32') for set in sets])
        dataset = tf.data.Dataset.from_tensor_slices(\
            data_slices).shuffle(len(data_slices[0]), reshuffle_each_iteration=True).batch(self.batch_size)
        return dataset

    def log_loss(self, act_true, pred_dis):
        likelihood = pred_dis.log_prob(act_true)
        # tf.print(tf.reduce_min(likelihood))
        # tf.print(tf.reduce_mean(likelihood))
        # tf.print(tf.shape(likelihood))
        # tf.print(tf.reduce_mean(likelihood))
        return -tf.reduce_mean(likelihood)
