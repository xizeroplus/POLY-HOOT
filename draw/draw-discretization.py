import os
import _tkinter
from matplotlib import pyplot as plt
import numpy as np



simulations = [20, 60, 100]
colors = ['#ff7f00', '#f781bf', '#4daf4a',
                  '#377eb8', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


x = [2 * i for i in range(1, 11)]

fig, ax = plt.subplots()

for id in range(len(simulations)):

	file = open('UCT-discretization-' + str(simulations[id]) + '.txt', 'r')
	lines = file.readlines()
	print(len(lines))
	file.close()

	a = [[] for j in range(10)]
	for i in range(10):
		for j in range(10):
			line = lines[i * 10 + j].split('\t')[1]
			a[j].append(float(line))
	a = np.array(a)


	err = []
	mean = []

	for j in range(10):
		tmp  = a[j]
		err.append(1.96 * np.std(tmp) / np.sqrt(len(tmp)))
		mean.append(np.mean(tmp))

	err = np.array(err)
	mean = np.array(mean)


	ax.plot(x, mean, color=colors[id], label=str(simulations[id]) + ' simulations')
	# ax.set_xscale('log')
	ax.fill_between(x, (mean - err), (mean + err), color=colors[id], alpha=.1)

ax.legend(loc='lower left', fontsize='x-large')
plt.xticks(np.arange(2, 21, step=3))
plt.ylim((50, 80))
plt.xlabel('Number of discretized actions', fontsize='x-large')
plt.ylabel('Rewards', fontsize='x-large')
plt.savefig('UCT-discretization.pdf')
plt.show()