import numpy as np 
import sys 

def rt_gen(inp_file):

	file_obj = open(inp_file, "r")
	file_lines = file_obj.readlines()

	i = 0
	states = 0
	actions = 0

	for f in file_lines:
		if i == 0:
			states = int(f)
		elif i == 1:
			actions = int(f)
		i += 1
		# print f

	# print states, actions

	rewards = np.zeros((states, actions, states), dtype=float)
	transitions = np.zeros((states, actions, states), dtype=float)

	i = 0
	j = 0
	k = 0
	for f in file_lines:
		state_num = ((i-2)/(actions))%states
		action_num = (i-2)%actions
		if i >= 2 and i < states*actions + 2:
			f_split = f.split()
			for index, value in enumerate(f_split):
				rewards[int(state_num), int(action_num), int(index)] = value
		elif i >= states*actions + 2 and i < 2*(states*actions) + 2:
			f_split = f.split()
			for index, value in enumerate(f_split):
				transitions[int(state_num), int(action_num), int(index)] = value
		elif i == 2*(states*actions) + 2:
			gamma = float(f)
		elif i == 2*(states*actions) + 3:
			type_mdp = f
		i += 1

	# print rewards
	# print transitions
	# print gamma
	# print type_mdp

	return states, actions, rewards, transitions, gamma, type_mdp