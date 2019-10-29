import numpy as np 
from parse import parse
import sys 
from pulp import *

def model_basedRL(inp_file):
	[state_trajectory, action_trajectory, reward_trajectory, S, A, gamma] = parse(inp_file)

	T = np.zeros((S, A, S), dtype=float)
	R = np.zeros((S, A, S), dtype=float)
	prob_actions = np.zeros((S, A))
	total_trans = np.zeros((S, A, S))
	total_reward = np.zeros((S, A, S))
	total_visits = np.zeros((S, A))

	state = state_trajectory[0]
	action = action_trajectory[0]
	reward = reward_trajectory[0]

	for x in range(1,len(state_trajectory)):
		next_state = state_trajectory[x]

		total_trans[state][action][next_state] += 1
		total_visits[state][action] += 1
		total_reward[state][action][next_state] += reward

		if x != len(state_trajectory) - 1:
			state = state_trajectory[x]
			action = action_trajectory[x]
			reward = reward_trajectory[x]

	for s in range(0, S):
		visits_s = np.sum(total_visits[s])
		for a in range(0, A):
			if visits_s != 0:
				prob_actions[s][a] = float(total_visits[s][a])/visits_s
			else:
				prob_actions[s][a] = 0
			visits_s_a = np.sum(total_trans[s][a])
			for ns in range(0, S):
				if total_trans[s][a][ns] != 0:
					R[s][a][ns] = float(total_reward[s][a][ns])/total_trans[s][a][ns]
				else:
					R[s][a][ns] = 0
				if visits_s_a != 0:
					T[s][a][ns] = float(total_trans[s][a][ns])/visits_s_a
				else:
					T[s][a][ns] = 0

	val_function = pulp.LpProblem("Finding the Value Function")
	state_nums = [p for p in range(0, S)]
	v_s = pulp.LpVariable.dicts('qpi', state_nums, cat='Continuous')

	for s in range(0, S):
		val_function += v_s[s] == pulp.lpSum( (prob_actions[s][a] * pulp.lpSum( (T[s, a, s_dash]*( R[s, a, s_dash] + gamma * v_s[s_dash] ) ) for s_dash in range(0, S)) ) for a in range(0, A) )

	val_function.solve()

	value_functions = np.zeros(S, dtype=np.float128)
	for s in range(0, S):
		value_functions[s] = pulp.value(v_s[s])
		print (value_functions[s])

	value_functions_d1 = np.asarray([66.2637, 72.967])
	value_functions_d2 = np.asarray([0.644577, 0.728381, 0.80263, 0.716817, 0.580511, 0.561383])
	
	# mse_d1 = (value_functions - value_functions_d1)**2
	# mse_d1 = np.sum(mse_d1) 
	# print (mse_d1)

	# mse_d2 = (value_functions - value_functions_d2)**2
	# mse_d2 = np.sum(mse_d2)
	# print (mse_d2)

def td_lambda(inp_file, lambda_, alpha_0):
	[state_trajectory, action_trajectory, reward_trajectory, S, A, gamma] = parse(inp_file)

	value_functions = np.zeros(S, dtype=np.float128)

	eligibility_trace = np.zeros(S, dtype=float)

	state = state_trajectory[0]
	action = action_trajectory[0]
	reward = reward_trajectory[0]

	alpha = alpha_0

	for x in range(1,len(state_trajectory)):
		next_state = state_trajectory[x]

		delta = reward + gamma*value_functions[next_state] - value_functions[state]
		eligibility_trace[state] += 1
		for p in range(0,S):
			value_functions[p] = value_functions[p] + alpha*delta*eligibility_trace[p] 

		eligibility_trace = [gamma*lambda_*eligibility_trace[s] for s in range(0, S)]

		alpha = float(alpha_0)/(x+1)

		if x != len(state_trajectory)-1:
			state = state_trajectory[x]
			action = action_trajectory[x]
			reward = reward_trajectory[x]

	for x in range(0,S):
		print (value_functions[x])

	value_functions_d1 = np.asarray([66.2637, 72.967])
	value_functions_d2 = np.asarray([0.644577, 0.728381, 0.80263, 0.716817, 0.580511, 0.561383])
	
	# mse_d1 = (value_functions - value_functions_d1)**2
	# mse_d1 = np.sum(mse_d1) 
	# print (mse_d1)

	mse_d2 = (value_functions - value_functions_d2)**2
	mse_d2 = np.sum(mse_d2)
	print (mse_d2)

if __name__ == '__main__':
	inp_file = sys.argv[1]
	lambda_ = 1
	alpha = 9
	model_basedRL(inp_file)
	# td_lambda(inp_file, lambda_, alpha)
