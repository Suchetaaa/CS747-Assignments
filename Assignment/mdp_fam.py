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
discounts_1 = np.arange(0.01, 0.39, 0.1)
discounts_2 = np.arange(0.41, 0.74, 0.1)
discounts_3 = np.arange(0.76, 0.99, 0.1)

# print discounts_1
# print discounts_2
# print discounts_3

curr_policy_1 = np.zeros(states, dtype=int)
new_policy_1 = np.zeros(states, dtype=int)
curr_policy_2 = np.zeros(states, dtype=int)
new_policy_2 = np.zeros(states, dtype=int)
curr_policy_3 = np.zeros(states, dtype=int)
new_policy_3 = np.zeros(states, dtype=int)
transitions = np.zeros((states, actions), dtype=float)
rewards = np.zeros((states, actions), dtype=float)

k = 0
while True:
	done_1 = 1
	done_2 = 1
	done_3 = 1
	print "Iteration {}".format(k)
	transitions = np.random.uniform(transition_lower, transition_upper, (states, actions, states))
	rewards = np.random.uniform(reward_lower, reward_higher, (states, actions, states))
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
		break

	k += 1



