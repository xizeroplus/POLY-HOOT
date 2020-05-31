from __future__ import print_function
from __future__ import division

import os
import numpy as np
import random


INF = 1e9

def evaluate_single_point(x):
	return -(x[0] - 0.3)**2  -(x[1] - 0.7)**2 + 1


def flip(p):
	return True if random.random() < p else False

class HOO_node(object):
	def __init__(self,cell,value,height,dimension,num):

		'''This is a node of the MFTREE
		cell: tuple denoting the bounding boxes of the partition
		m_value: mean value of the observations in the cell and its children
		value: value in the cell
		fidelity: the last fidelity that the cell was queried with
		upp_bound: B_{i,t} in the paper
		t_bound: upper bound with the t dependent term
		height: height of the cell (sometimes can be referred to as depth in the tree)
		dimension: the dimension of the parent that was halved in order to obtain this cell
		num: number of queries inside this partition so far
		left,right,parent: pointers to left, right and parent
		'''
		self.cell = cell
		self.m_value = value
		self.value = value
		self.height = height
		self.dimension = dimension
		self.num = num
		self.t_bound = 0

		self.left = None
		self.right = None
		self.parent = None


def in_cell(node,parent):
	'''
	Check if 'node' is a subset of 'parent'
	node can either be a MF_node or just a tuple denoting its cell
	'''
	try:
		ncell = list(node.cell)
	except:
		ncell = list(node)
	pcell = list(parent.cell)
	flag = 0

	for i in range(len(ncell)):
		if ncell[i][0] >= pcell[i][0] and ncell[i][1] <= pcell[i][1]:
			flag = 0
		else:
			flag = 1
			break
	if flag == 0:
		return True
	else:
		return False


class HOO_tree(object):
	'''
	MF_tree class that maintains the multi-fidelity tree
	nu: nu parameter in the paper
	rho: rho parameter in the paper
	sigma: noise variance, ususally a hyperparameter for the whole process
	C: parameter for the bias function as defined in the paper
	root: can initialize a root node, when this parameter is supplied by a MF_node object instance
	'''
	def __init__(self, nu, rho, root=None):
		self.nu = nu
		self.rho = rho
		self.root = root
		self.root = root
		self.mheight = 0
		self.maxi = float(-INF)
		self.current_best = root
		self.last_leaf = None 
	

	def update_parents(self, node, val):
		if node is None:
			return
		else:
			node.m_value = (node.num * node.m_value + val)/(1.0 + node.num)
			node.num += 1.0
			# node.upp_bound = node.m_value + 2 * ((self.rho)**(node.height)) * self.nu
			self.update_parents(node.parent,val)


	def update_tbounds(self,root,t):
		'''
		updating the tbounds of every node recursively
		'''
		if root is None:
			return
		self.update_tbounds(root.left, t)
		self.update_tbounds(root.right, t)
		root.t_bound = root.m_value + 2 * ((self.rho)**(root.height)) * self.nu + np.sqrt(2 * np.log(t) / root.num)
		maxi = None
		if root.left:
			maxi = root.left.t_bound
		if root.right:
			if maxi:
				if maxi < root.right.t_bound:
					maxi = root.right.t_bound
			else:
				maxi = root.right.t_bound
		if maxi:
			root.t_bound = min(root.t_bound,maxi)


	def print_given_height(self,root,height):
		if root is None:
			return
		if root.height == height:
			print (root.cell, root.num, root.t_bound),
		elif root.height < height:
			if root.left:
				self.print_given_height(root.left,height)
			if root.right:
				self.print_given_height(root.right,height)
		else:
			return


	def levelorder_print(self):
		'''
		levelorder print
		'''
		for i in range(self.mheight + 1):
			self.print_given_height(self.root,i)
			print('\n')


	def search_cell(self,root,cell):
		'''
		check if a cell is present in the tree
		'''
		if root is None:
			return False,None,None
		if root.left is None and root.right is None:
			if root.cell == cell:
				return True,root,root.parent
			else:
				return False,None,root
		if root.left:
			if in_cell(cell,root.left):
				return self.search_cell(root.left,cell)
		if root.right:
			if in_cell(cell,root.right):
				return self.search_cell(root.right,cell)


	def get_next_node(self,root):
		'''
		getting the next node to be queried or broken, see the algorithm in the paper
		'''
		if root is None:
			print('Could not find next node. Check Tree.')
		
		if root.left is None and root.right is None:
			bit = flip(0.5)
			if bit:
				return root, 0 
			else:
				return root, 1 
		if root.left is None:
			return root, 0
		if root.right is None:
			return root, 1 

		if root.left.t_bound > root.right.t_bound:
			return self.get_next_node(root.left)
		elif root.left.t_bound < root.right.t_bound:
			return self.get_next_node(root.right)
		else:
			bit = flip(0.5)
			if bit:
				return self.get_next_node(root.left)
			else:
				return self.get_next_node(root.right)


	def get_current_best(self,root):
		'''
		get current best cell from the tree
		'''
		if root is None:
			return
		if root.right is None and root.left is None:
			val = root.m_value - self.nu * ((self.rho)**(root.height))
			if self.maxi < val:
				self.maxi = val 
				cell = list(root.cell) 
				self.current_best =np.array([(s[0]+s[1])/2.0 for s in cell])
			return
		if root.left:
			self.get_current_best(root.left)
		if root.right:
			self.get_current_best(root.right)





class HOO(object):
	'''
	MFHOO algorithm, given a fixed nu and rho
	mfobject: multi-fidelity noisy function object
	nu: nu parameter
	rho: rho parameter
	budget: total budget provided either in units or time in seconds
	sigma: noise parameter
	C: bias function parameter
	tol: default parameter to decide whether a new fidelity query is required for a cell
	Randomize: True implies that the leaf is split on a randomly chosen dimension, False means the scheme in DIRECT algorithm is used. We recommend using False.
	Auto: Select C automatically, which is recommended for real data experiments
	CAPITAL: 'Time' mean time in seconds is used as cost unit, while 'Actual' means unit cost used in synthetic experiments
	debug: If true then more messages are printed
	'''
	def __init__(self,dim, nu, rho):
		self.nu = nu
		self.rho = rho
		self.t = 0
		cell = tuple([(0,1)]*dim)
		height = 0
		dimension = 0
		# diam = nu * (rho**height)
		# bhi = 2 * diam
		root = HOO_node(cell, 0, height, dimension, 0)
		self.Tree = HOO_tree(nu, rho, root)
		self.Tree.root.t_bound = INF


	def get_value(self,cell):
		'''cell: tuple'''
		x = np.array([(s[0]+s[1])/2.0 for s in list(cell)])
		return x #, evaluate_single_point(x)


	def querie(self, cell, height, rho, nu,dimension):
		# diam = nu * (rho**height)
		action = self.get_value(cell)
	
		# bhi = 2*diam
		current_object = HOO_node(cell, 0, height, dimension, 0)
		return action, current_object


	def split_children(self, parent, rho, nu, child_id):
		pcell = list(parent.cell)
		span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]

		dimension = np.argmax(span)
		dd = len(pcell)
		if dimension == parent.dimension:
			dimension = (parent.dimension - 1)%dd
		h = parent.height + 1
		l = np.linspace(pcell[dimension][0],pcell[dimension][1],3)
		
		cell = []
		for j in range(len(pcell)):
			if j != dimension:
				cell = cell + [pcell[j]]
			else:
				cell = cell + [(l[child_id],l[child_id + 1])]
		cell = tuple(cell)
		action, child = self.querie(cell, h, rho, nu,dimension)

		return action, child


	def select_action(self):
		parent, child_id = self.Tree.get_next_node(self.Tree.root)
		action, current = self.split_children(parent, self.rho, self.nu, child_id)
		if child_id == 0:
			parent.left = current
			parent.left.parent = parent 
			self.Tree.last_leaf = parent.left
		else:
			parent.right = current
			parent.right.parent = parent 
			self.Tree.last_leaf = parent.right

		return action 



	def update(self, value):
		self.t = self.t + 1
		self.Tree.update_parents(self.Tree.last_leaf, value)
		self.Tree.update_tbounds(self.Tree.root, self.t)


	def run(self, iters):
		for _ in range(iters):
			action = self.select_action()
			value = evaluate_single_point(action)
			self.update(value)


	def get_point(self):
		self.Tree.get_current_best(self.Tree.root)
		return self.Tree.current_best


if __name__ == '__main__':
	dim = 2
	rho = 2**(-2 / dim)
	nu = 4 * dim
	HOO_iters = 1000
	hoo = HOO(dim=dim, nu=nu, rho=rho)
	hoo.run(HOO_iters)
	print(hoo.get_point())
