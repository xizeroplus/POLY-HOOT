from __future__ import print_function
from __future__ import division
from math import log, sqrt
import gym
from SnapshotENV import SnapshotEnv
import pickle
import os
import numpy as np
from itertools import count
from hoo import HOO
import random



envname = 'Continuous-CartPole-v0'
# envname = 'Pendulum-v0'
env = gym.make(envname).env
KEY_DECIMAL = 4
MAX_MCTS_DEPTH = 100
ITERATIONS = 200
TEST_ITERATIONS = 500
discount = 0.99
INF = 1e9

if envname == 'Continuous-CartPole-v0':
	min_action = env.min_action
	max_action = env.max_action
	dim = 1
elif envname == 'Pendulum-v0':
	min_action = -2.0
	max_action = 2.0
	dim = 1

env = SnapshotEnv(gym.make(envname).env)


class Node:
	# parent = None
	# value_sum = 0
	# times_visited = 0


	def __init__(self, snapshot, obs, is_done, parent, dim):
		self.parent = parent
		self.snapshot = snapshot
		self.obs = obs
		self.is_done = is_done
		# self.is_leaf = True
		self.children = {}
		self.immediate_reward = 0
		self.dim = dim
		
		# res = env.get_result(self.parent.snapshot, action)
		# self.snapshot, self.obs, self.immediate_reward, self.is_done, _ = res
		# self.children = set()
		
		rho = 2**(-2 / dim)
		nu = 4 * dim
		self.hoo = HOO(dim=dim, nu=nu, rho=rho, min_value=min_action, max_value=max_action)
		

	# def __repr__(self):
	# 	return f'Node({self.parent}, {self.action})'


	# def is_root(self):
	# 	return self.parent is None


	def selection(self, depth):
		if self.is_done or depth > MAX_MCTS_DEPTH:
			return 0
		raw_action = self.hoo.select_action().tolist()
		action = [round(a, KEY_DECIMAL) for a in raw_action]
		# if len(action) == 1:
		# 	action = action[0]
		if tuple(action) in self.children:
			child = self.children[tuple(action)]
			immediate_reward = child.immediate_reward
			value = child.selection(depth + 1)
			self.hoo.update(value + immediate_reward)
			return immediate_reward + value
		else:
			snapshot, obs, immediate_reward, is_done, _ = env.get_result(self.snapshot, action)
			child = Node(snapshot, obs, is_done, self, self.dim)
			child.immediate_reward = immediate_reward
			self.children[tuple(action)] = child 
			value = child.selection(depth + 1)
			self.hoo.update(value + immediate_reward)
			return immediate_reward + value


	# def expand(self):
	# 	# if self.times_visited == 0:
	# 	# 	return self
	# 	for j in range(n_actions):
	# 		self.children.add(Node(self, [discretized_actions[j]]))
	# 	assert not self.is_done, "the episode is finished! can't expand"

	# 	return self.selection()


	# def rollout(self, t_max=500):
	# 	if self.is_done:
	# 		return 0.
	# 	env.load_snapshot(self.snapshot)
	# 	total_reward_rollout = 0
	# 	for _ in range(t_max):
	# 		action = np.array([random.choice(discretized_actions)])
	# 		# action = env.action_space.sample()
	# 		next_s, r, done, _ = env.step(action)
	# 		total_reward_rollout += r
	# 		if done: break
	# 	return total_reward_rollout


	# def back_propagate(self, rollout_reward):
	# 	node_value = self.immediate_reward + rollout_reward
	# 	self.value_sum += node_value
	# 	self.times_visited += 1

	# 	if not self.is_root():
	# 		self.parent.back_propagate(rollout_reward)


	# def safe_delete(self):
	# 	"""
	# 	for deleting unnecessary node
	# 	"""
	# 	del self.parent
	# 	for child in self.children:
	# 		child.safe_delete()
	# 		del child


def plan_mcts(root, n_iter):
	"""
	builds tree with mcts for n_iter
	root: tree root to plan from
	n_iter: how many select->expand->rollout->back_propagate
	"""
	for _ in range(n_iter):
		value = root.selection(depth=0)
		
		# if node.is_done:
		# 	node.back_propagate(0)
		# else:
		# 	best_leaf = node.expand()
		# 	rollout_reward = best_leaf.rollout()
		# 	best_leaf.back_propagate(rollout_reward)


if __name__ == '__main__':
	env = SnapshotEnv(gym.make(envname).env)
	root_obs = env.reset()
	root_snapshot = env.get_snapshot()
	root = Node(root_snapshot, root_obs, False, None, dim)
	current_discount = 1.0

	plan_mcts(root, n_iter=ITERATIONS)

	test_env = pickle.loads(root_snapshot) # env used to show progress
	total_reward = 0
	for i in range(TEST_ITERATIONS):

		print(i)
		raw_best_action = root.hoo.get_point().tolist()
		best_action = np.array([round(a, KEY_DECIMAL) for a in raw_best_action])
		# if len(best_action) == 1:
		# 	best_action = best_action[0]

		s, r, done, _ = test_env.step(best_action)
		
		test_env.render()
		
		total_reward += r * current_discount
		current_discount *= discount
		print(total_reward)
		if done:
			print(f"finished with reward: {total_reward}")
			test_env.close()
			break
		
		# delete other actions

		root = root.children[tuple(best_action)]
		plan_mcts(root, n_iter=ITERATIONS)
	
	test_env.close()
	print(total_reward)



