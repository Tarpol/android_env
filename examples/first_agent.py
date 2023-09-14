# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script demonstrating usage of AndroidEnv."""

import numpy as np
import tensorflow as tf
import random
import time
from android_env import loader
from dm_env import specs
from collections import deque

from absl import app
from absl import logging
from absl import flags
FLAGS = flags.FLAGS

# Simulator args.
flags.DEFINE_string('avd_name', None, 'Name of AVD to use.')
flags.DEFINE_string('android_avd_home', '~/.android/avd', 'Path to AVD.')
flags.DEFINE_string('android_sdk_root', '~/Android/Sdk', 'Path to SDK.')
flags.DEFINE_string('emulator_path',
                    '~/Android/Sdk/emulator/emulator', 'Path to emulator.')
flags.DEFINE_string('adb_path',
                    '~/Android/Sdk/platform-tools/adb', 'Path to ADB.')
flags.DEFINE_bool('run_headless', False,
                  'Whether to display the emulator window.')

# Environment args.
flags.DEFINE_string('task_path', None, 'Path to task textproto file.')

# Experiment args
# Sum of Steps
flags.DEFINE_integer('num_steps', 10000, 'Number of steps to take.')

# Define the DQN agent
class DQNAgent:
    def __init__(self, observation_space, action_spec: dict):
        self.observation_space = observation_space
        self.action_spec = action_spec
        self.action_size = sum([v.num_values for v in self.action_spec.values() if isinstance(v, specs.DiscreteArray)])
        # TODO: Take care of parameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        # Epsilon should start at 1 on an untrained model
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

        # TODO: Legacy Code - Not used since output is from different bodies
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten(input_shape=self.observation_space))
        # model.add(tf.keras.layers.Dense(24, activation='relu'))
        # model.add(tf.keras.layers.Dense(24, activation='relu'))
        # model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        # return model

    # Builds the NN
    def _build_model(self):
        input_layer = tf.keras.layers.Input(shape=self.observation_space)
        x = tf.keras.layers.Flatten()(input_layer)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        
        # Q-values for action type
        action_type_output = tf.keras.layers.Dense(3, activation='linear', name='action_type')(x)  # 3 possible actions
        
        # Q-values for touch positions
        touch_position_output = tf.keras.layers.Dense(2, activation='sigmoid', name='touch_position')(x)  # x and y
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=[action_type_output, touch_position_output])
        model.compile(loss=['mse', 'mse'], optimizer=tf.keras.optimizers.Adam(), loss_weights=[1.0, 1.0])
        
        return model

    def act(self, state) -> dict[str, np.ndarray]:
      action = {}
      if np.random.rand() <= self.epsilon:
        
        # Discovery actions, mutation
        action_type_spec = self.action_spec['action_type']
        action['action_type'] = np.random.randint(
            low=action_type_spec.minimum, 
            high=action_type_spec.maximum + 1,  # inclusive upper bound
            dtype=action_type_spec.dtype
        )

        # Decide on the touch_position. As with action_type, in a real agent, this decision
        # would be based on some policy. Here we're just choosing a random position.
        touch_position_spec = self.action_spec['touch_position']
        action['touch_position'] = np.random.uniform(
            low=touch_position_spec.minimum, 
            high=touch_position_spec.maximum,
            size=touch_position_spec.shape
        ).astype(touch_position_spec.dtype)
        
        return action
      
      q_values = self.model.predict(state)
      action_type_predicted = np.argmax(q_values[0])  
      touch_position_predicted = q_values[1]  # since it's already in [x, y] format

      action['action_type'] = action_type_predicted
      action['touch_position'] = touch_position_predicted
      return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main(_):

  with loader.load(
      emulator_path=FLAGS.emulator_path,
      android_sdk_root=FLAGS.android_sdk_root,
      android_avd_home=FLAGS.android_avd_home,
      avd_name=FLAGS.avd_name,
      adb_path=FLAGS.adb_path,
      task_path=FLAGS.task_path,
      run_headless=FLAGS.run_headless) as env:

    observation_space = env.observation_spec()['pixels'].shape
    action_spec = env.action_spec()
    agent = DQNAgent(observation_space, action_spec)

    def get_random_action() -> dict[str, np.ndarray]:
      """Returns a random AndroidEnv action."""
      action = {}
      for k, v in action_spec.items():
        if isinstance(v, specs.DiscreteArray):
          action[k] = np.random.randint(low=0, high=v.num_values, dtype=v.dtype)
        else:
          action[k] = np.random.random(size=v.shape).astype(v.dtype)
      return action

    timestep = env.reset()

    for step in range(FLAGS.num_steps):
      state_pixels = np.expand_dims(timestep.observation['pixels'], axis=0)
      action = agent.act(state_pixels)
      print("action start")
      print(action)
      print("action end")

      # action = get_random_action()
      print(action['touch_position'])
      action['touch_position'] = action['touch_position'][0]
      print(action['touch_position'])
      timestep = env.step(action)
      time.sleep(0.01)
      reward = timestep.reward
      logging.info('Step %r, action: %r, reward: %r', step, action, reward)


if __name__ == '__main__':
  logging.set_verbosity('info')
  logging.set_stderrthreshold('info')
  flags.mark_flags_as_required(['avd_name', 'task_path'])
  app.run(main)
