import os
import _tkinter
from matplotlib import pyplot as plt
import numpy as np

task_name = 'Continuous-CartPole-v0.txt'
# task_name = 'Continuous-CartPole-IG-v0.txt'
# alg_names = ['discretized-UCT', 'HOOT', 'POLY-HOOT']
alg_names = ['discretized-UCT', 'PUCT', 'HOOT', 'POLY-HOOT']
colors = ['#ff7f00', '#f781bf', '#4daf4a',
                  '#377eb8', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

TESTS = 14

base = 333.3333 ** (1.0 / 15.0)
x = [int(3* (base ** i)) for i in range(14)]

fig, ax = plt.subplots()

for alg_id in range(len(alg_names)):

	file = open(alg_names[alg_id] + task_name, 'r')
	lines = file.readlines()
	print(len(lines))
	file.close()

	a = [[] for j in range(TESTS)]
	for i in range(10):
		for j in range(TESTS):
			line = lines[i * TESTS + j].split('\t')[1]
			a[j].append(float(line))
	a = np.array(a)


	err = []
	mean = []

	for j in range(TESTS):
		tmp  = a[j]
		err.append(1.96 * np.std(tmp) / np.sqrt(len(tmp)))
		mean.append(np.mean(tmp))

	err = np.array(err)
	mean = np.array(mean)


	ax.plot(x, mean, color=colors[alg_id], label=alg_names[alg_id])
	ax.set_xscale('log')
	ax.fill_between(x, (mean - err), (mean + err), color=colors[alg_id], alpha=.1)

ax.legend(loc='lower right', fontsize='x-large')
plt.xlabel('Rounds of simulations', fontsize='x-large')
plt.ylabel('Rewards', fontsize='x-large')
plt.savefig(task_name.split('.')[0] + '.pdf')
plt.show()