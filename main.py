import gym
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


env = gym.make('CartPole-v0')
episodes =10
episode_length = 200
print(env.observation_space.shape)

# TODO: more generic model function
# TODO: Step 1: +/- reinforcement depending on result compared to baseline. Need to store all steps data + exploration randomness
def nn_model(input_size,output_size, learning_rate):
    model = Sequential()
    model.add(Dense(12, input_dim=input_size, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
    return model



for t in range(episodes):
    observation = env.reset()
    for s in range(episode_length):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

