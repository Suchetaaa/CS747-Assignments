import numpy as np 
import random 
from lp_solve import lp

transition_lower = 0
transition_upper = 1
reward_lower = -1
reward_higher = 1
states = 2
actions = 2
type_mdp = "continuous"
discounts_1 = np.arange(0.01, 0.39, 0.001)
discounts_2 = np.arange(0.41, 0.74, 0.001)
discounts_3 = np.arange(0.76, 0.99, 0.001)

# print discounts_1
# print discounts_2
# print discounts_3

curr_policy_1 = np.zeros(states, dtype=int)
new_policy_1 = np.zeros(states, dtype=int)
curr_policy_2 = np.zeros(states, dtype=int)
new_policy_2 = np.zeros(states, dtype=int)
curr_policy_3 = np.zeros(states, dtype=int)
new_policy_3 = np.zeros(states, dtype=int)
transitions = np.array([[[0.84938371, 0.26502715],
							[0.39455493, 0.2298523]],
							
							[[0.11407442, 0.96699579],
							[ 0.83935457, 0.86356795]]])
rewards = np.array([[[ 0.31078263, -0.17806394],
						[0.97522669, -0.18964429]],
							
						[[0.98402895, 0.83044068],
						[-0.10980786, -0.35387885]]])

k = 0
while k == 0:
	done_1 = 1
	done_2 = 1
	done_3 = 1
	print "Iteration {}".format(k)
	# transitions = np.random.uniform(transition_lower, transition_upper, (states, actions, states))
	# rewards = np.random.uniform(reward_lower, reward_higher, (states, actions, states))
	print transitions
	print rewards
	for i, p in enumerate(discounts_1):
		new_policy_1 = lp(states, actions, rewards, transitions, p, type_mdp)
		if i == 0 or (new_policy_1 == curr_policy_1).all():
			curr_policy_1 = new_policy_1
		else:
			done_1 = 0
			break

	if done_1 == 1:
		for i, p in enumerate(discounts_2):
			new_policy_2 = lp(states, actions, rewards, transitions, p, type_mdp)
			if i == 0 or (new_policy_2 == curr_policy_2).all():
				curr_policy_2 = new_policy_2
			else:
				done_2 = 0
				break

	if done_1 == 1 and done_2 == 1:
		for i, p in enumerate(discounts_3):
			new_policy_3 = lp(states, actions, rewards, transitions, p, type_mdp)
			if i == 0 or (new_policy_3 == curr_policy_3).all():
				curr_policy_3 = new_policy_3
			else:
				done_3 = 0
				break

	if done_1 == 1 and done_2 == 1 and done_3 == 1 and not((new_policy_1 == new_policy_2).all()) and not((new_policy_2 == new_policy_3).all()) and not((new_policy_1 == new_policy_3).all()):
		print "Success"
		break

	k += 1



