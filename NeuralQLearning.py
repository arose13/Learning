import gym
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import
import keras

model = Sequential()
model.fit()
model.evaluate()


NUM_ACTIONS = 2
NUM_STATES = 4
MAX_REPLAY_STATES = 100
BATCH_SIZE = 20
N_TRAINING_GAMES = 1000
JUMP_FPS = 1  # was 2


class Brain(object):
    def __init__(self):
        self.model = None


    def create_model(self):
        # I know I could have just given it a list but I like making the lists this way
