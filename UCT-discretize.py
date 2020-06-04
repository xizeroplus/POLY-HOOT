from math import log, sqrt
import gym
import copy
from SnapshotENV import SnapshotEnv
import pickle
import os
import numpy as np
from itertools import count
import random

envname = 'Continuous-CartPole-v0'
# envname = 'Acrobot-v1'
# envname = 'LunarLanderContinuous-v2'
# envname = 'Pendulum-v0'

env = gym.make(envname).env

filename = 'UCT' + envname + '.txt'

MAX_MCTS_DEPTH = 50
ITERATIONS = 100
TEST_ITERATIONS = 150
n_actions = 10
dim = 1

if envname == 'Continuous-CartPole-v0':
	min_action = env.min_action
	max_action = env.max_action
	discretized_actions = [min_action + i * (max_action - min_action) / (n_actions - 1) for i in range(n_actions)]
elif envname == 'Pendulum-v0':
	min_action = -2.0
	max_action = 2.0
	discretized_actions = [min_action + i * (max_action - min_action) / (n_actions - 1) for i in range(n_actions)]
elif envname == 'LunarLanderContinuous-v2':
	dim = 2
	min_action = -1.0
	max_action = 1.0
	MAX_MCTS_DEPTH = 100
	k = int(np.sqrt(n_actions))
	arr = np.linspace(min_action, max_action, k + 1)
	discretized_actions = []
	for x in arr:
		for y in arr:
			discretized_actions.append([x, y])
	n_actions = len(discretized_actions)

env = SnapshotEnv(gym.make(envname).env)

SCALE_UCB = 10
MAX_VALUE = 1e100
discount = 0.99

class Node:
	parent = None
	value_sum = 0
	times_visited = 0

	def __init__(self, parent, action):
		self.parent = parent
		self.action = action
		self.children = set()
		res = env.get_result(self.parent.snapshot, action)
		self.snapshot, self.obs, self.immediate_reward, self.is_done, _ = res
		

	def __repr__(self):
		return f'Node({self.parent}, {self.action})'


	def is_root(self):
		return self.parent is None


	def is_leaf(self):
		return len(self.children) == 0


	def get_mean_value(self):
		return self.value_sum / self.times_visited if self.times_visited != 0 else 0.


	def ucb_score(self):
		U = 2 * sqrt(log(self.parent.times_visited) / self.times_visited) if self.times_visited != 0 else MAX_VALUE
		return self.get_mean_value() + SCALE_UCB * U


	def selection(self):
		if self.is_leaf():
			return self
		children = list(self.children)
		best_leaf = children[np.argmax([child.ucb_score() for child in children])]
		return best_leaf.selection()


	def expand(self):
		# if self.times_visited == 0:
		# 	return self
		for j in range(n_actions):
			if dim == 1:
				self.children.add(Node(self, [discretized_actions[j]]))
			else:
				self.children.add(Node(self, discretized_actions[j]))
		assert not self.is_done, "the episode is finished! can't expand"

		return self.selection()


	def rollout(self, t_max=MAX_MCTS_DEPTH):
		if self.is_done:
			return 0.
		env.load_snapshot(self.snapshot)
		total_reward_rollout = 0
		for _ in range(t_max):
			if dim == 1:
				action = np.array([random.choice(discretized_actions)])
			else:
				action = np.array(discretized_actions[np.random.randint(0, n_actions)])
			# action = env.action_space.sample()
			next_s, r, done, _ = env.step(action)
			total_reward_rollout += r
			if done: break
		return total_reward_rollout


	def back_propagate(self, rollout_reward):
		node_value = self.immediate_reward + rollout_reward
		self.value_sum += node_value
		self.times_visited += 1

		if not self.is_root():
			self.parent.back_propagate(rollout_reward)


	def safe_delete(self):
		"""
		for deleting unnecessary node
		"""
		del self.parent
		for child in self.children:
			child.safe_delete()
			del child


class Root(Node):
	"""
	creates special node that acts like tree root
    snapshot: snapshot (from env.get_snapshot) to start planning from
    observation: last environment observation
	"""
	def __init__(self, snapshot, obs):

		self.parent = self.action = None
		self.snapshot = snapshot
		self.obs = obs
		self.children = set()
		self.ucb_score = 0
		self.immediate_reward = 0
		self.is_done = False


	@staticmethod
	def to_root(node):
		"""initializes node as root"""
		root = Root(node.snapshot, node.obs)
		attr_names = ["value_sum", "times_visited",
		"children", "is_done"]
		for attr in attr_names:
			setattr(root, attr, getattr(node, attr))
		return root


def plan_mcts(root, n_iter):
	"""
	builds tree with mcts for n_iter
	root: tree root to plan from
	n_iter: how many select->expand->rollout->back_propagate
	"""
	for _ in range(n_iter):
		node = root.selection()
		if node.is_done:
			node.back_propagate(0)
		else:
			best_leaf = node.expand()
			rollout_reward = best_leaf.rollout()
			best_leaf.back_propagate(rollout_reward)


env = gym.make(envname).env
env = SnapshotEnv(gym.make(envname).env)
root_obs_ori = env.reset()
root_snapshot_ori = env.get_snapshot()

base = 333.3333 ** (1.0 / 15.0)
samples = [int(3* (base ** i)) for i in range(16)]

if __name__ == '__main__':
	for ITERATIONS in samples[0:-3]:
		root_obs = copy.copy(root_obs_ori)
		root_snapshot = copy.copy(root_snapshot_ori)
		root = Root(root_snapshot, root_obs)
		current_discount = 1.0

		plan_mcts(root, n_iter=ITERATIONS)

		test_env = pickle.loads(root_snapshot) # env used to show progress
		total_reward = 0
		for i in range(TEST_ITERATIONS):

			print(i)
			children = list(root.children)
			best_child = children[np.argmax([child.get_mean_value() for child in children])]

			s, r, done, _ = test_env.step(best_child.action)
			
			# test_env.render()
			total_reward += r * current_discount
			current_discount *= discount
			print(total_reward)
			if done:
				file = open('ori-' + filename, 'a')
				file.write(str(ITERATIONS) + '\t' + str(total_reward) + '\n')
				file.close()
				print(f"finished with reward: {total_reward}")
				test_env.close()
				break
			# no need for the other part of the tree because the role of
			# root will be transmited to the best_child
			for child in children:
				if child != best_child:
					child.safe_delete()

			root = Root.to_root(best_child)
			# if root.is_leaf():
			plan_mcts(root, n_iter=ITERATIONS)

			# best_leaf = root.expand()
			# child_value = root.rollout()
			# root.back_propagate(child_value)
		if not done:
			file = open('ori-' + filename, 'a')
			file.write(str(ITERATIONS) + '\t' + str(total_reward) + '\n')
			file.close()
			print(total_reward)
			test_env.close()



