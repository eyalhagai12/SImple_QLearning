import random

from GridEnv import GridEnv
from QTable import QTable


def main():
    grid_size = 15
    env = GridEnv(grid_size)
    state_shape = (grid_size, grid_size, grid_size, grid_size)
    table = QTable(4, state_shape, env.get_actions(), lr=0.1, discount_factor=0.2)

    # define training parameters
    episodes = 50000
    max_steps = 25
    exploration_prob = 0.5
    test_episodes = 20
    train_episodes = episodes - (test_episodes + 1)
    n_notify = 1000

    for episode_num in range(episodes):
        # init new episode
        if episode_num % n_notify == 0:
            print("Starting Episode ", episode_num)

        # reset environment
        obs = env.reset()

        # step parameters
        steps = 0
        done = False
        test = episode_num > train_episodes

        if episode_num == train_episodes:
            print("Starting Test Episodes")

        # main while loop for each episode
        while not done and steps < max_steps:
            prob = random.random()
            agent_location = obs["agent"]
            target_location = obs["target"]

            # make a decision (either explore or exploit)
            if prob <= exploration_prob and not test:
                # exploration
                action = random.randint(0, 3)
            else:
                # exploitation
                action = table.get_next_move(agent_location, target_location)

            # do the step and update Q-table
            obs, reward, done, _ = env.step(action)

            if not test:
                table.update(agent_location, target_location, action, reward)

            steps += 1

            # show progression every n_notify episodes
            if test:
                env.render()
    print("Done!")


if __name__ == '__main__':
    main()
