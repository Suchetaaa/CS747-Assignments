import numpy as np 
import numpy.random as random
import sys
import os

def roundRobin(num_bandits, bandit_probs, epsilon, horizon, seed):

	random.seed(seed)

	rewards = np.zeros((num_bandits, horizon), dtype=int)

	for y in xrange(0,num_bandits): 
		s = np.random.binomial(1, bandit_probs[y], horizon)
		rewards[y, :] = s 

	cum_reward = 0
	# cum_reward_horizons = np.array([0, 0, 0, 0, 0, 0])

	curr_arm = 0
	pulls = np.zeros(num_bandits, dtype=int)

	for y in xrange(0,horizon):
		curr_reward = rewards[curr_arm, pulls[curr_arm]]
		pulls[curr_arm] += 1
		cum_reward += curr_reward
		curr_arm = (curr_arm + 1)%num_bandits

		# if y == 49:
		# 	cum_reward_horizons[0] = cum_reward
		# elif y == 199:
		# 	cum_reward_horizons[1] = cum_reward
		# elif y == 799:
		# 	cum_reward_horizons[2] = cum_reward
		# elif y == 3199:
		# 	cum_reward_horizons[3] = cum_reward
		# elif y == 12799:
		# 	cum_reward_horizons[4] = cum_reward
		# elif y == 51199:
		# 	cum_reward_horizons[5] = cum_reward
		# else: continue

	# print cum_reward
	return cum_reward

if __name__ == '__main__':

	instance = sys.argv[1]
	epsilon = float(sys.argv[2])
	horizon = int(sys.argv[3])
	seed = int(sys.argv[4])

	instance_path = instance[3:]

	abs_path = os.path.abspath(__file__)  
	present_dir = os.path.dirname(abs_path)
	parent_dir = os.path.dirname(present_dir)
	path = os.path.join(parent_dir, instance_path)

	file_instance = open(path, "r")
	f_lines = file_instance.readlines()
	a = []
	for h in f_lines:
		a.append(float(h.strip()))
	bandit_probs = np.array(a)

	num_bandits = (bandit_probs.shape)[0]
	# print num_bandits
	max_p = np.amax(bandit_probs)
	# print max_p

	curr_reward = roundRobin(num_bandits, bandit_probs, epsilon, horizon, seed)
	regret = max_p*horizon - curr_reward

	# file_obj = open("outputData.txt", "a")
	# horizons = np.array([50, 200, 800, 3200, 12800, 51200])

	# regrets_horizons = max_p*horizons - curr_reward_horizons

	# for x in xrange(0,6):
	# 	file_obj.write("../instances/i-{}.txt, round-robin, {}, {}, {}, {}\n".format(instance+1, seed, epsilon, horizons[x], regrets_horizons[x]))
	# file_obj.write("../instances/i-{}.txt, round-robin, {}, {}, {}, {}\n".format(instance+1, seed, epsilon, horizon, regret))

	# file_obj.close()

	# file_obj = open("outputs1.txt", "a")
	print "{}, round-robin, {}, {}, {}, {}\n".format(instance, seed, epsilon, horizon, regret)
	# file_obj.close()
	
		

