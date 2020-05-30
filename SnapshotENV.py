from gym.core import Wrapper
import pickle
from collections import namedtuple

ActionResult = namedtuple('actionResult',
	('snapshot', 'next_state', 'reward', 'is_done', 'info'))

class SnapshotEnv(Wrapper):
	"""
	Creates a wrapper that supports saving and loading environemnt states.
    Required for planning algorithms.
	"""
	def get_snapshot(self, render=False):
		"""
		returns: environment state that can be loaded with load_snapshot 
        Snapshots guarantee same env behaviour each time they are loaded.
		"""
		if render:
			self.render()
			self.close()

		if self.unwrapped.viewer:
			self.unwrapped.viewer.close()
			self.unwrapped.viewer = None

		return pickle.dumps(self.env)


	def load_snapshot(self, snapshot, render=False):
		"""
		loads snapshot as current env state
		"""
		assert not hasattr(self, "_monitor") or hasattr(
			self.env, "_monitor"), "can't backtrack while recording"

		if render:
			self.render()
			self.close()

		self.env = pickle.loads(snapshot)


	def get_result(self, parent_snapshot, parent_action):
		"""
		store result from a (parent-snapshot, action-parent)
		"""
		self.load_snapshot(parent_snapshot)
		next_s, r, is_done, info = self.step(parent_action)
		node_snapshot = self.get_snapshot()
		res = ActionResult(snapshot=node_snapshot,
			next_state=next_s, reward=r, is_done=is_done, info=info)

		return res

"""
Testing

if __name__ == '__main__':
	import gym
	env = SnapshotEnv(gym.make('CartPole-v0'))
	print(env.reset())
	print(env.action_space.n)
	snap = env.get_snapshot()
	res = env.get_result(snap, 0)
	print(res)

"""