import numpy as np


class QTable:
    def __init__(self, n_actions: int, state_shape: tuple, action_dict, lr=0.01, discount_factor=0.4):
        self.q_table = np.zeros((*state_shape, n_actions))
        self.action_dict = action_dict
        self.lr = lr
        self.discount = discount_factor

        # show q values
        self.show_q_values = False

    def get_next_move(self, agent_location, target_location):
        if self.show_q_values:
            print(self.q_table[agent_location[0], agent_location[1], target_location[0], target_location[1], :])
        return np.argmax(self.q_table[agent_location[0], agent_location[1], target_location[0], target_location[1], :])

    def update(self, agent_location, target_location, action, reward):
        temp_agent = agent_location + self.action_dict[action]
        optimal_future_value = 0
        try:
            optimal_future_value = self.get_next_move(temp_agent, target_location)
        except Exception as e:
            pass

        current_value = self.q_table[
            agent_location[0], agent_location[1], target_location[0], target_location[1], action]

        # update weight
        self.q_table[
            agent_location[0], agent_location[1], target_location[0], target_location[
                1], action] = current_value + self.lr * (
                reward + self.discount * optimal_future_value - current_value)
