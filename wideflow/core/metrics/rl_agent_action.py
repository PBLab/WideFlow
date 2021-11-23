from core.abstract_metric import AbstractMetric
import cupy as cp
import pickle


class DQNAgentAction(AbstractMetric):
    def __init__(self, input, agent, training_mode, roi_pixels_list, capacity, ptr):
        self.input = input
        self.agent = agent

        self.training_mode = training_mode  # define if session is used also to train the DQN solver
        self.pretrained = agent.pretrained
        self.total_reward = 0
        self.total_rewards = 0

        self.capacity = capacity
        if len(agent.state_space) > 2:
            if agent.state_space[0] + 1 < capacity:
                self.state_depth = agent.state_space[0]
            else:
                raise ValueError(f'agent state space depth must be equal or smaller than input depth + 1'
                                 f'got space depth: {agent.state_space[0]}, input depth: {self.capacity}')
        else:
            self.state_depth = 1
        self.state = cp.ndarray(self.agent.state_space, dtype=self.input.dtype)
        self.prev_state = cp.ndarray(self.agent.state_space, dtype=self.input.dtype)

        self.roi_pixels_list = roi_pixels_list

        self.ptr = ptr

    def initialize_buffers(self):
        if self.training_mode and self.pretrained:
            with open("total_rewards.pkl", 'rb') as f:
                self.total_rewards = pickle.load(f)

        self.ptr = self.capacity - 1
        self.prev_state[:] = cp.roll(self.input, -(self.ptr + 1))[-self.state_depth:]
        self.prev_reward = self.evaluate_state_reward()
        self.prev_action = self.agent.act(self.prev_reward)

    def evaluate(self):
        if self.ptr == self.capacity - 1:
            self.ptr = 0
        else:
            self.ptr += 1

        self.state[:] = cp.roll(self.input, -(self.ptr + 1))[-self.state_depth:]
        action = self.agent.act(self.state)
        reward = self.evaluate_state_reward()

        self.total_reward += reward

        if self.training_mode:  # add to replay memory in a delay fashion of 1 frame since we got a stochastic world
            self.agent.remember(self.prev_state, self.prev_action, self.prev_reward, self.state)
            self.agent.experience_replay()

        self.prev_state[:] = cp.asnumpy(self.state)
        self.prev_action = action
        self.prev_reward = reward

        return action

    def evaluate_state_reward(self):
        return cp.asnumpy(cp.mean(self.x[self.ptr, self.pixels_inds[1], self.pixels_inds[0]]))
