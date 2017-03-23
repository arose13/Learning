import gym
import copy
import random
import numpy as np
from itertools import count
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.losses import mse
from keras import backend as K


NUM_ACTIONS = 2
NUM_STATES = 4
MAX_REPLAY_STATES = 100
BATCH_SIZE = 20
N_TRAINING_GAMES = 1000
JUMP_FPS = 1  # was 2

def huber_loss(target, prediction):
    return K.mean(K.sqrt(1 + K.square(prediction - target)) - 1, axis=-1)


class Brain(object):
    GAMMA = 0.9  # decay rate
    EPSILON = 1  # Exploration
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.05
    LEARNING_RATE = 1e-2

    def __init__(self, model: Model, environment: gym.Env):
        self.model = model  # Keras model
        self.env = environment
        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n
        self.memory = deque(maxlen=int(1e6))

    def act(self, state):
        if np.random.rand() <= self.EPSILON:
            return self.env.action_space.sample()
        action = self.model.predict(state)
        return np.argmax(action[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32*4):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        x = np.zeros((batch_size, self.state_size))
        y = np.zeros((batch_size, self.action_size))

        for i, sample in enumerate(minibatch):
            state, action, reward, next_state, done = sample
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.GAMMA * np.amax(self.model.predict(next_state)[0])

            x[i], y[i] = state, target

        self.model.fit(x, y, batch_size=batch_size, epochs=1, verbose=0)
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY


if __name__ == '__main__':
    game_env = gym.make('CartPole-v0')
    n_episodes = 1000
    penalty_points = -10

    # Model
    dnn = Sequential()
    dnn.add(Dense(20, input_dim=game_env.observation_space.shape[0], activation='tanh'))

    dnn.add(Dense(20, activation='tanh'))
    # dnn.add(Dropout(0.5))

    dnn.add(Dense(game_env.action_space.n, activation='linear'))
    dnn.compile(
        # loss=mse,
        loss=huber_loss,
        # optimizer=Adam()
        optimizer=RMSprop(lr=Brain.LEARNING_RATE)
        # optimizer=Adam(lr=Brain.LEARNING_RATE)
    )
    dnn.summary()

    # Player
    player = Brain(dnn, game_env)

    # Play Game
    for episode in range(n_episodes):
        # init
        observation = game_env.reset()
        observation = np.reshape(observation, [1, 4])

        # NOTE This differs from the example
        for score in count(1):
            action = player.act(observation)

            # Only show every 100 games
            # if episode % 100 == 0:
            #     game_env.render()

            next_observation, reward, done, _ = game_env.step(action)
            next_observation = np.reshape(observation, [1, 4])

            # reward is always one. So we'll convert this to huge penalty and the game ends
            reward = penalty_points if done else 1

            player.remember(observation, action, reward, next_observation, done)
            observation = copy.deepcopy(next_observation)

            # Check Termination Condition
            if done:
                print('Episode: {} of {} | Score: {}'.format(
                    episode, n_episodes, score
                ))
                break

            # Have the player think about what it's learnt
            player.replay()
