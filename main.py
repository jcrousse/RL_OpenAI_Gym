import gym
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import numpy as np

from collections import deque


class minimal_RL_model:
    def __init__(self, env):

        self.env = env


        #exploration/exploitation parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

        self.model = self.nn_model(0.005)

        self.best_reward = 0

    def explor_exploit(self, state):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def nn_model(self, learning_rate):

        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n

        print(output_size)

        model = Sequential()
        model.add(Dense(12, input_dim=input_size, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(12, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, activation="softmax"))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
        return model

    def shape_state(self, state):
        return state.reshape(1, self.env.observation_space.shape[0])

    def train_model(self, total_reward, X, Y, epochs=5):

        # print("current epsilon: ", self.epsilon)

        if total_reward >30:

            self.best_reward = total_reward
            print("new best score: ", self.best_reward)

            X = X.reshape(int(X.shape[0]/4), 4)
            Y = Y.reshape(int(Y.shape[0]/2), 2)

            self.model.fit(X[0:-5], Y[0:-5], epochs, verbose=0)
        else:
            self.epsilon = 1
            # increase e^silon ?
            # always train but discard last examples?

    def predict(self, state):
        return np.argmax(self.model.predict(state)[0])

def main():

    env = gym.make('CartPole-v0')
    print(env.observation_space.shape)

    episodes = 1000
    episode_length = 200

    learning_model = minimal_RL_model(env)

    for t in range(5):
        observation = learning_model.shape_state(env.reset())
        total_reward = 0
        for s in range(episode_length):

            action = learning_model.predict(observation)
            observation, reward, done, info = env.step(action)
            observation = learning_model.shape_state(observation)
            if done:
                print("Test episode{0} timesteps: {1}".format(t, s + 1))
                break

    for t in range(episodes):
        observation = learning_model.shape_state(env.reset())

        memory_X = np.array([])
        memory_Y = np.array([])

        total_reward = 0

        for s in range(episode_length):

            #env.render()
            #print(observation)

            action = learning_model.explor_exploit(observation) # action = 0 or 1 for CartPole.

            # convert 0 or 1 action to one-hot [1,0] or [0,1]
            target_action = np.zeros(env.action_space.n)
            target_action[action] = 1


            observation, reward, done, info = env.step(action)

            total_reward =total_reward + reward

            memory_X = np.append(memory_X, observation)
            memory_Y = np.append(memory_Y, target_action)

            observation = learning_model.shape_state(observation)

            if done:
                print("Episode {0} finished after {1} timesteps".format(t, s+1))
                break

        learning_model.train_model(total_reward, memory_X, memory_Y)

    for t in range(5):
        observation = learning_model.shape_state(env.reset())
        total_reward = 0
        for s in range(episode_length):

            action = learning_model.predict(observation)
            observation, reward, done, info = env.step(action)
            observation = learning_model.shape_state(observation)
            if done:
                print("Test episode{0} timesteps: {1}".format(t, s + 1))
                break


if __name__ == "__main__":
    main()