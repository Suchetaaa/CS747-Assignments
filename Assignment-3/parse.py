import sys
import numpy as np 

def parse(inp_file):

	file_obj = open(inp_file, "r")
	file_lines = file_obj.readlines()

	i = 0
	states = 0
	actions = 0
	discount_factor = 0
	state_trajectory = []
	action_trajectory = []
	reward_trajectory = []

	for f in file_lines:
		if i == 0:
			states = int(f)
		elif i == 1:
			actions = int(f)
		elif i == 2:
			discount_factor = float(f)
		else:
			f_split = f.split()
			if len(f_split) == 3:
				state_trajectory.append(f_split[0])
				action_trajectory.append(f_split[1])
				reward_trajectory.append(f_split[2])
			else:
				state_trajectory.append(f_split[0])
		i += 1

	state_trajectory = np.asarray(state_trajectory, dtype=int)
	action_trajectory = np.asarray(action_trajectory, dtype=int)
	reward_trajectory = np.asarray(reward_trajectory, dtype=float)

	return state_trajectory, action_trajectory, reward_trajectory, states, actions, discount_factor

if __name__ == '__main__':
	inp_file = "/Users/suchetaaa/Desktop/Academics @IITB/Semester VII/FILA/FILA-Assignments/Assignment-3/data/d1.txt"
	parse(inp_file)