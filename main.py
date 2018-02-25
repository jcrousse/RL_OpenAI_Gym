import gym

from keras.models import Sequential
from keras.layers import Dense, Dropout


import numpy as np


class minimal_RL_model:
    def __init__(self, env):

        self.env = env

        #exploration/exploitation parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

        self.model = self.nn_model()

        self.best_reward = 0

        # self.initial_epoch = 0

    def explor_exploit(self, state):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def nn_model(self, learning_rate=0.005):

        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n

        model = Sequential()

        model.add(Dense(32, input_dim=input_size, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, activation="softmax"))

        model.compile(loss="mean_squared_error", optimizer="adam")

        return model

    def shape_state(self, state):
        return state.reshape(1, self.env.observation_space.shape[0])

    def train_model(self, total_reward, X, Y, epochs=5):

        if total_reward >40:

            self.best_reward = total_reward
            print("new best score: ", self.best_reward)

            X = X.reshape(int(X.shape[0]/4), 4)
            Y = Y.reshape(int(Y.shape[0]/2), 2)

            self.model.fit(X[0:-10], Y[0:-10], epochs, verbose=0)
        else:
            self.epsilon = 1
            # increase e^silon ?
            # always train but discard last examples?

    def predict(self, state):
        return np.argmax(self.model.predict(state)[0])

    def get_weights(self):
        return self.model.get_weights()

    def model_fit(self, X, Y, epochs = 5):
        self.model.fit(X,Y, epochs = epochs, verbose=0) #  initial_epoch=self.initial_epoch
        # self.initial_epoch += epochs


class EnvRunner:
    def __init__(self, env):
        self.env = env

        self.observations_batch = []
        self.actions_batch = []
        self.scores = []

        self.batch_data = []

    def run(self, num_episodes, predict_model=None, num_steps=200, cut_last_steps=0):
        # batch_data = []
        for _ in range(num_episodes):

            observations_episode, actions_episode, score = self.run_episode(predict_model, num_steps, cut_last_steps)

            self.observations_batch.append(np.array(observations_episode))
            self.actions_batch.append(np.array(actions_episode))
            self.scores.append(score)

        return self

    def run_episode(self, predict_model, num_steps, cut_last_steps):
        observation = self.env.reset()
        score = 0
        observations_episode, actions_episode = [], []
        for step in range(num_steps):
            num_actions = self.env.action_space.n
            action = np.random.randint(0, num_actions)
            if isinstance(predict_model, minimal_RL_model):
                action = predict_model.predict(observation.reshape(1, 4))
            one_hot_action = np.zeros(num_actions)
            one_hot_action[action] = 1
            observations_episode.append(observation)
            actions_episode.append(one_hot_action)

            observation, reward, done, _ = self.env.step(action)
            score += reward
            if done:
                break
        # return observations_episode[:-cut_last_steps], actions_episode[:-cut_last_steps], score
        if cut_last_steps > len(observations_episode):
            observations_episode=observations_episode[:-cut_last_steps]
            actions_episode = actions_episode[:-cut_last_steps]
        return observations_episode, actions_episode, score

    def trim_batch(self, success_threshold):

        self.observations_batch = self.trim_array(self.observations_batch, self.scores, success_threshold)
        self.actions_batch = self.trim_array(self.actions_batch, self.scores, success_threshold)
        self.scores = self.trim_array(self.scores, self.scores, success_threshold)

        return self

    def trim_array(self, base_array, condition_array, threshold):
        return [item for idx, item in enumerate(base_array) if condition_array[idx] > threshold]

    def get_run_data(self):
        # print(type(self.scores))
        if len(self.scores) > 0:
            return np.concatenate(self.observations_batch), np.concatenate(self.actions_batch), self.scores
        else:
            # raise Exception('No data to return !')  # alternatively raise exception if no data
            return np.array([]), np.array([]), self.scores

    def get_scores(self):
        return self.scores

    def reset(self):
        self.observations_batch = []
        self.actions_batch = []
        self.scores = []
        return self


def main():

    env = gym.make('CartPole-v0')

    env_runner = EnvRunner(env)
    learning_model = minimal_RL_model(env)

    batches = [5000,1000,400,200,50,50,50,50]
    # training seems "anchored" on results of first training round (if keeping track of epoch)
    # maybe optimizer to be adjusted for data size.

    for batch_size in batches:

        print("gathering {0} episodes of train data".format(batch_size))
        trainx, trainy, scores = env_runner.run(num_episodes=batch_size, cut_last_steps=10).trim_batch(30).get_run_data()

        if len(scores)>0:
            print("data gathered, training model")
            learning_model.model_fit(trainx, trainy, epochs=5)
            print("model trained.")
        else:
            print("no data to train on")

        print("running model on environment")
        _, _, scores = env_runner.reset().run(predict_model=learning_model, num_episodes=50).get_run_data()
        print("average score: ", np.mean(scores))

    # training on a few more example to check if performances are affected:


if __name__ == "__main__":
    main()