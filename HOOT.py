from math import log, sqrt
import gym
from SnapshotENV import SnapshotEnv
import pickle
import os
import numpy as np
import copy
from hoo import HOO
import random



envname = 'Continuous-CartPole-v0'
# envname = 'Pendulum-v0'
# envname = 'LunarLanderContinuous-v2'
env = gym.make(envname).env
KEY_DECIMAL = 4
MAX_MCTS_DEPTH = 50
ITERATIONS = 100
TEST_ITERATIONS = 150
discount = 0.99
INF = 1e9
filename = 'HOOT' + envname + '.txt'

if envname == 'Continuous-CartPole-v0':
	min_action = env.min_action
	max_action = env.max_action
	dim = 1
elif envname == 'Pendulum-v0':
	min_action = -2.0
	max_action = 2.0
	dim = 1
elif envname == 'LunarLanderContinuous-v2':
	MAX_MCTS_DEPTH = 100
	min_action = -1.0
	max_action = 1.0
	dim = 2

env = SnapshotEnv(gym.make(envname).env)


class Node:
	def __init__(self, snapshot, obs, is_done, parent, dim):
		self.parent = parent
		self.snapshot = snapshot
		self.obs = obs
		self.is_done = is_done
		self.children = {}
		self.immediate_reward = 0
		self.dim = dim
		
		rho = 2**(-2 / dim)
		nu = 4 * dim
		self.hoo = HOO(dim=dim, nu=nu, rho=rho, min_value=min_action, max_value=max_action)
		

	def selection(self, depth):
		if self.is_done or depth > MAX_MCTS_DEPTH:
			return 0
		raw_action = self.hoo.select_action().tolist()
		action = [round(a, KEY_DECIMAL) for a in raw_action]
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

	def delete(self, node):
		for action in node.children:
			node.delete(node.children[action])
		del node 



env = gym.make(envname).env
env = SnapshotEnv(gym.make(envname).env)
root_obs_ori = env.reset()
root_snapshot_ori = env.get_snapshot()

base = 333.3333 ** (1.0 / 15.0)
samples = [int(3* (base ** i)) for i in range(16)]

if __name__ == '__main__':
	for ITERATIONS in samples[0:-5]:
		root_obs = copy.copy(root_obs_ori)
		root_snapshot = copy.copy(root_snapshot_ori)
		root = Node(root_snapshot, root_obs, False, None, dim)
		current_discount = 1.0

		for _ in range(ITERATIONS):
			root.selection(depth=0)

		test_env = pickle.loads(root_snapshot)
		total_reward = 0
		for i in range(TEST_ITERATIONS):
			raw_best_action = root.hoo.get_point().tolist()
			best_action = np.array([round(a, KEY_DECIMAL) for a in raw_best_action])

			state, reward, done, _ = test_env.step(best_action)
			
			# test_env.render()
			
			total_reward += reward * current_discount
			current_discount *= discount
			print(i, total_reward)
			if done:
				file = open('ori-' + filename, 'a')
				file.write(str(ITERATIONS) + '\t' + str(total_reward) + '\n')
				file.close()
				print('ended with reward: ', total_reward)
				test_env.close()
				break
			
			# delete the other actions
			for action in root.children:
				if tuple(best_action) != action:
					root.delete(root.children[action])

			root = root.children[tuple(best_action)]
			for _ in range(ITERATIONS):
				root.selection(depth=0)

		if not done: 
			test_env.close()
			
			file = open('ori-' + filename, 'a')
			file.write(str(ITERATIONS) + '\t' + str(total_reward) + '\n')
			file.close()
			print(total_reward)



