import tensorflow as tf
from tensorflow.keras import layers

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, param_dim):
        super(QNetwork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.q_value = layers.Dense(action_dim, activation='linear')
        self.param_value = layers.Dense(param_dim)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q_value = self.q_value(x)
        param_value = self.param_value(x)
        return q_value, param_value

class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dim, param_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.param = layers.Dense(param_dim, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        param = self.param(x)
        return param