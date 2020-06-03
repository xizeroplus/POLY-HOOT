import os
import numpy as np
import random


INF = 1e9

# used for testing
def evaluate_single_point(x):
	# return -(x[0] + 0.5)**2  -(x[1] - 0.7)**2 + 1
	return -(x[0] + 0.3)**2  + 1



class HOO_node(object):
	def __init__(self,cell,value,height,dimension,num):

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


class POLY_HOO_tree(object):

	def __init__(self, nu, rho, alpha, xi, eta, root=None, lim_depth=10):
		self.nu = nu
		self.rho = rho
		self.alpha = alpha
		self.xi = xi
		self.eta = eta 
		self.root = root
		self.lim_depth = lim_depth
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
			self.update_parents(node.parent,val)


	def update_tbounds(self,root,t):

		if root is None:
			return
		self.update_tbounds(root.left, t)
		self.update_tbounds(root.right, t)
		root.t_bound = root.m_value + 2 * ((self.rho)**(root.height)) * self.nu + 0.1 * t**(self.alpha / self.xi) * root.num**(self.eta - 1.0)
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



	def get_next_node(self,root):
		if root is None:
			print('Could not find next node. Check Tree.')
		
		if root.left is None and root.right is None:
			if random.random() < 0.5:
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
			if random.random() < 0.5:
				return self.get_next_node(root.left)
			else:
				return self.get_next_node(root.right)


	def get_current_best(self,root):
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



class POLY_HOO(object):

	def __init__(self,dim, nu, rho, min_value, max_value, lim_depth, alpha, xi, eta):
		self.nu = nu
		self.rho = rho
		self.alpha = alpha
		self.xi = xi
		self.eta = eta
		self.t = 0
		self.lim_depth = lim_depth
		cell = tuple([(min_value, max_value)] * dim)
		height = 0
		dimension = 0
		root = HOO_node(cell, 0, height, dimension, 0)
		self.Tree = POLY_HOO_tree(nu, rho, alpha, xi, eta, root, lim_depth)
		self.Tree.root.t_bound = INF


	def get_value(self,cell):
		x = np.array([(s[0]+s[1])/2.0 for s in list(cell)])
		return x


	def querie(self, cell, height, rho, nu,dimension):
		action = self.get_value(cell)
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
		if parent.height >= self.lim_depth:
			action = self.get_value(parent.cell)
			self.Tree.last_leaf = parent
			return action
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
	dim = 1
	rho = 2**(-2 / dim)
	nu = 4 * dim
	HOO_iters = 1000
	lim_depth = 10
	alpha = 5
	xi = 10
	eta = 0.5
	poly_hoo = POLY_HOO(dim=dim, nu=nu, rho=rho, min_value=-1.0, max_value=1.0, lim_depth=lim_depth, alpha=alpha, xi=xi, eta=eta)
	poly_hoo.run(HOO_iters)
	print(poly_hoo.get_point())
