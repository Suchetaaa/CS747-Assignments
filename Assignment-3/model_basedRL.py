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


if __name__ == '__main__':
	inp_file = "/Users/suchetaaa/Desktop/Academics @IITB/Semester VII/FILA/FILA-Assignments/Assignment-3/data/d2.txt"
	model_basedRL(inp_file)
