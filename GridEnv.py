from typing import Optional, Union, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.core import ObsType


def generate_random_pos(size):
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    return np.array([x, y])


class GridEnv(gym.Env):
    def __init__(self, size):
        # create grid
        self.grid = np.zeros((size, size))
        self.grid_size = size
        self.target_positions = np.zeros_like(self.grid)

        # number of steps taken to reach goal
        self.n_steps = 0

        # create observation space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
        })

        # create action space
        self.action_space = spaces.Discrete(4)

        # define possible actions
        self._actions = {
            0: np.array([0, 1]),  # right
            1: np.array([0, -1]),  # left
            2: np.array([1, 0]),  # down
            3: np.array([-1, 0]),  # up
        }

        # for pycharm to not cry
        self._agent_location = None
        self._target_location = None
        self._agent_prev_location = None

        # define color map for plotting
        self.color_map = {
            0: (0, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255)
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location)}

    def get_actions(self) -> dict:
        return self._actions

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # call the super class
        super().reset(seed=seed)

        # init grid
        self.grid = np.zeros_like(self.grid)

        # initiate agent and target
        self._agent_location = generate_random_pos(self.grid_size)
        self._target_location = generate_random_pos(self.grid_size)
        while all(np.equal(self._target_location, self._agent_location)):
            self._target_location = generate_random_pos(self.grid_size)

        # count target positions
        self.target_positions[self._target_location[0]][self._target_location[1]] += 1

        # put them on the grid
        self.grid[self._agent_location[0]][self._agent_location[1]] = 1  # agent is 1 on the grid
        self.grid[self._target_location[0]][self._target_location[1]] = 2  # target is 2 on the grid

        # reset steps
        self.n_steps = 0

        # return observation and info
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def render(self, mode="human"):
        # show positions
        grid_image = np.array(list(map(lambda x: self.color_map[x], self.grid.flatten()))).reshape(
            (self.grid_size, self.grid_size, 3)).astype(np.uint8)

        plt.cla()
        plt.imshow(grid_image)
        plt.pause(0.1)

    def step(self, action):
        # set a reward variable to track
        reward = 0
        last_distance = self._get_info()["distance"]

        # move agent in given direction
        self.grid[self._agent_location[0]][self._agent_location[1]] = 0
        next_position = self._agent_location + self._actions[action]

        # increase steps
        self.n_steps += 1

        # check if we attempt to go out of boundary
        if all(next_position >= np.zeros((2,))) and all(next_position < np.array([self.grid_size, self.grid_size])):
            self._agent_location = next_position
            current_distance = self._get_info()["distance"]
            reward = last_distance - current_distance

        # check if agent is done
        done = all(np.equal(self._agent_location, self._target_location))

        # put agent back on board
        if not done:
            self.grid[self._agent_location[0]][self._agent_location[1]] = 1
        else:
            reward = 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info
