import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from memory import ReplayBuffer
import os
from tensorflow.keras.utils import plot_model
from model import QNetwork, ActorNetwork

class PDQNAgent:
    def __init__(self, state_dim, action_dim, param_dim, gamma=0.75, tau=0.005, buffer_size=20000, batch_size=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim, param_dim)
        self.target_q_network = QNetwork(state_dim, action_dim, param_dim)
        self.actor_network = ActorNetwork(state_dim, param_dim)
        self.target_actor_network = ActorNetwork(state_dim, param_dim)
        self._counter_steps=0
        self._update_freq=100

        self.q_optimizer = optimizers.Adam(learning_rate=1e-3)
        self.actor_optimizer = optimizers.Adam(learning_rate=1e-3)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_target_network(self.target_q_network, self.q_network, tau=1.0)
        self.update_target_network(self.target_actor_network, self.actor_network, tau=1.0)

    def update_target_network(self, target, source, tau):
        for target_param, source_param in zip(target.trainable_variables, source.trainable_variables):
            target_param.assign(tau * source_param + (1.0 - tau) * target_param)

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_dim)
            param = np.random.uniform(-1, 1, self.param_dim)
        else:
            q_values, _ = self.q_network(np.array([state]))
            action = np.argmax(q_values)
            param = self.actor_network(np.array([state]))[0].numpy()
        return action, param

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, params = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        params = np.array(params)

        # Train Q-network
        with tf.GradientTape() as tape:
            q_values, param_values = self.q_network(states, training=True)
            q_values = tf.reduce_sum(tf.one_hot(actions, self.action_dim) * q_values, axis=1)
            next_q_values, _ = self.target_q_network(next_states)
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            q_loss = tf.reduce_mean(tf.square(q_values - target_q_values))

        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))

        # Train Actor-network
        with tf.GradientTape() as tape:
            action_probs = self.actor_network(states, training=True)
            sampled_actions = tf.random.categorical(tf.math.log(action_probs), 1)
            sampled_actions = tf.squeeze(sampled_actions, axis=-1)
            action_log_probs = tf.reduce_sum(tf.math.log(action_probs) * tf.one_hot(sampled_actions, depth=self.action_dim), axis=1)

            # Calculate advantages
            q_values, _ = self.q_network(states, training=True)
            baseline_values = tf.reduce_sum(tf.one_hot(sampled_actions, self.action_dim) * q_values, axis=1)
            advantages = baseline_values - tf.reduce_mean(baseline_values)

            actor_loss = -tf.reduce_mean(action_log_probs * advantages)

        actor_grads = tape.gradient(actor_loss, self.actor_network.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))
        self._counter_steps+=1
        if self._counter_steps%self._update_freq==0:
            self.update_target_network(self.target_q_network, self.q_network, self.tau)
            self.update_target_network(self.target_actor_network, self.actor_network, self.tau)

    def add_experience(self, state, action, reward, next_state, done, param):
        self.replay_buffer.add((state, action, reward, next_state, done, param))

    def save_model(self, path):
        self.q_network.save(os.path.join(path, 'trained_model'), save_format='tf')
        plot_model(self.q_network, to_file=os.path.join(path, 'qnet_structure.png'), show_shapes=True, show_layer_names=True)
        self.actor_network.save(os.path.join(path, 'trained_actor'), save_format='tf')
        plot_model(self.actor_network, to_file=os.path.join(path, 'actor_structure.png'), show_shapes=True, show_layer_names=True)