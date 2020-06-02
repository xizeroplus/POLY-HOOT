from math import log, sqrt
import gym
import math
from SnapshotENV import SnapshotEnv
import pickle
import os
import numpy as np
from itertools import count
import random

envname = 'Continuous-CartPole-v0'
# envname = 'Pendulum-v0'
# envname = 'LunarLanderContinuous-v2'

env = gym.make(envname).env

filename = 'PW' + envname + '.txt'

MAX_MCTS_DEPTH = 25
ITERATIONS = 500
TEST_ITERATIONS = 150
n_actions = 1
dim = 1
alpha = 0.5

if envname == 'Continuous-CartPole-v0':
	min_action = env.min_action
	max_action = env.max_action
	discretized_actions = [np.random.uniform(min_action, max_action)]
elif envname == 'Pendulum-v0':
	min_action = -2.0
	max_action = 2.0
	discretized_actions = [np.random.uniform(min_action, max_action)]
elif envname == 'LunarLanderContinuous-v2':
	dim = 2
	min_action = -1.0
	max_action = 1.0
	MAX_MCTS_DEPTH = 100
	discretized_actions = [[np.random.uniform(min_action, max_action), np.random.uniform(min_action, max_action)]]

env = SnapshotEnv(gym.make(envname).env)

SCALE_UCB = 1
MAX_VALUE = 1e100
discount = 0.99

def new_action():
	if dim == 1:
		return np.random.uniform(min_action, max_action)
	elif dim == 2:
		return [np.random.uniform(min_action, max_action), np.random.uniform(min_action, max_action)]


class Node:
	parent = None
	times_visited = 0
	value_sum = 0
	
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
		if math.floor((self.times_visited + 1)**alpha) > math.floor((self.times_visited)**alpha):
			if dim == 1:
				child = Node(self, [new_action()])
			else:
				child = Node(self, new_action())
			self.children.add(child)
			return child
		children = list(self.children)
		best_leaf = children[np.argmax([child.ucb_score() for child in children])]
		return best_leaf.selection()


	def expand(self):
		# if self.times_visited == 0:
		# 	return self
		# for j in range(self.n_actions):
		# 	if dim == 1:
		# 		self.children.add(Node(self, [discretized_actions[j]]))
		# 	else:
		# 		self.children.add(Node(self, discretized_actions[j]))
		# assert not self.is_done, "the episode is finished! can't expand"
		if dim == 1:
			self.children.add(Node(self, [new_action()]))
		else:
			self.children.add(Node(self, new_action()))

		return self.selection()


	def rollout(self, t_max=MAX_MCTS_DEPTH):
		if self.is_done:
			return 0.
		env.load_snapshot(self.snapshot)
		total_reward_rollout = 0
		for _ in range(t_max):
			if dim == 1:
				action = env.action_space.sample()
				# action = np.array([random.choice(discretized_actions)])
			else:			
				action = env.action_space.sample()
				# action = np.array(discretized_actions[np.random.randint(0, n_actions)])
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


if __name__ == '__main__':
	for test in range(10):
		env = gym.make(envname).env
		env = SnapshotEnv(gym.make(envname).env)
		root_obs = env.reset()
		root_snapshot = env.get_snapshot()
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
				file = open(filename, 'a')
				file.write(str(total_reward) + '\n')
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
			file = open(filename, 'a')
			file.write(str(total_reward) + '\n')
			file.close()
			print(total_reward)
			test_env.close()




