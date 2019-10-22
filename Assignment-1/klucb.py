import numpy as np 
import sys 
import numpy.random as random 
import os

def KL(p, q):
	if p == 1:
		return p*np.log(p/q)
	elif p == 0:
		return (1-p)*np.log((1-p)/(1-q))
	else:
		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def solve_q(rhs, p_a):
	if p_a == 1:
		return 1 
	q = np.arange(p_a, 1, 0.01)
	lhs = []
	for el in q:
		lhs.append(KL(p_a, el))
	lhs_array = np.array(lhs)
	lhs_rhs = lhs_array - rhs
	lhs_rhs[lhs_rhs <= 0] = np.inf
	min_index = lhs_rhs.argmin()
	return q[min_index]

def ucb_func(pulls, arm_rewards, time_steps, num_bandits):
	ucb_arms = np.zeros(num_bandits, dtype=float)
	for x in xrange(0,num_bandits):
		p_a = arm_rewards[x]/pulls[x]
		rhs = (np.log(time_steps) + 3*np.log(np.log(time_steps)))/pulls[x]
		ucb_arms[x] = solve_q(rhs, p_a)
	# print ucb_arms
	return ucb_arms

def kl_ucb(num_bandits, bandit_probs, epsilon, horizon, seed):
	
	random.seed(seed)

	rewards = np.zeros((num_bandits, horizon), dtype=int)

	for y in xrange(0,num_bandits): 
		s = np.random.binomial(1, bandit_probs[y], horizon)
		rewards[y, :] = s 

	cum_reward = 0
	# cum_reward_horizons = np.array([0, 0, 0, 0, 0, 0])

	curr_arm = 0
	curr_reward = 0
	pulls = np.zeros(num_bandits, dtype=int)
	arm_rewards = np.zeros(num_bandits, dtype=int)
	ucb_arms = np.zeros(num_bandits, dtype=float)

	for x in xrange(0,min(num_bandits,horizon)):
		# print x
		curr_arm = x
		curr_reward = rewards[curr_arm, pulls[curr_arm]]
		pulls[curr_arm] += 1
		cum_reward += curr_reward
		arm_rewards[curr_arm] += curr_reward
		
	if horizon > num_bandits:

		for y in xrange(num_bandits,horizon):
			
			ucb_arms = ucb_func(pulls, arm_rewards, y, num_bandits)
			max_ucb = np.amax(ucb_arms)
			indices = np.where(ucb_arms == max_ucb)
			
			curr_arm = np.amax(indices)
			curr_reward = rewards[curr_arm, pulls[curr_arm]]
			pulls[curr_arm] += 1
			cum_reward += curr_reward
			arm_rewards[curr_arm] += curr_reward
			# print "{}".format(curr_arm)
			# print y

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

	curr_reward = kl_ucb(num_bandits,bandit_probs, epsilon, horizon, seed)
	regret = max_p*horizon - curr_reward
	# print regret

	# file_obj = open("KLData.txt", "a")
	# horizons = np.array([50, 200, 800, 3200, 12800, 51200])

	# regrets_horizons = max_p*horizons - curr_reward_horizons

	# for x in xrange(0,6):
	# 	file_obj.write("../instances/i-{}.txt, kl-ucb, {}, {}, {}, {}\n".format(instance+1, seed, epsilon, horizons[x], regrets_horizons[x]))
	# file_obj.write("../instances/i-{}.txt, kl-ucb, {}, {}, {}, {}\n".format(instance+1, seed, epsilon, horizon, regret))

	# file_obj.close()

	# file_obj = open("outputs1.txt", "a")
	print "{}, kl-ucb, {}, {}, {}, {}\n".format(instance, seed, epsilon, horizon, regret)
	# file_obj.close()
                

