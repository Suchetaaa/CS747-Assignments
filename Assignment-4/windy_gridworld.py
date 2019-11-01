import numpy as np
import matplotlib.pyplot as plt 
import random

# actions: 
# 0 : up 
# 1 : down
# 2 : right 
# 3 : left

def nextstate_reward(num_rows, num_columns, wind_dist, start_block, goal_block, present_state, action):
	reward = -1
	wind_strength = wind_dist[int(present_state[1])]
	next_state = np.asarray([0, 0], dtype=int)

	if action == 0:
		next_state[0] = max(present_state[0] - 1 - wind_strength, 0)
		next_state[1] = max(present_state[1], 0)
	elif action == 1:
		next_state[0] = min(max(present_state[0] + 1 - wind_strength, 0), num_rows-1)
		next_state[1] = max(present_state[1], 0)
	elif action == 2:
		next_state[0] = max(present_state[0] - wind_strength, 0)
		next_state[1] = min(present_state[1] + 1, num_columns-1)
	elif action == 3:
		next_state[0] = max(present_state[0] - wind_strength, 0)
		next_state[1] = max(present_state[1] - 1, 0)

	if next_state[0] == goal_block[0] and next_state[1] == goal_block[1]:
		reward = 0

	return next_state, reward

def epsilon_greedy(present_state, epsilon, num_actions, Q):
	random1 = np.random.rand(1, 1)
	action = 0
	if random1 < epsilon: 
		action = random.randint(0, num_actions-1)
	else:
		action = np.amax(np.argmax(Q[present_state[0]][present_state[1]]))

	return action

def sarsa_on_policy(num_rows, num_columns, num_actions, num_episodes, wind_dist, start_block, goal_block, alpha, epsilon, num_runs, gamma):
	Q = np.zeros((num_rows, num_columns, num_actions), dtype=float)
	episodes_steps = np.zeros(num_episodes, dtype=float)

	for n in range(0, num_runs):
		print (n)
		for x in range(0,num_episodes):
			present_state = start_block
			action = epsilon_greedy(present_state, epsilon, num_actions, Q)
			time_steps = 0

			while(1):
				state_dash, reward = nextstate_reward(num_rows, num_columns, wind_dist, start_block, goal_block, present_state, action)
				action_dash = epsilon_greedy(state_dash, epsilon, num_actions, Q)
				Q[present_state[0]][present_state[1]][action] += alpha*(reward + gamma*Q[state_dash[0]][state_dash[1]][action_dash] - Q[present_state[0]][present_state[1]][action])
				present_state = state_dash
				action = action_dash
				time_steps += 1
				if present_state[0] == goal_block[0] and present_state[1] == goal_block[1]:
					break;
			episodes_steps[x] += time_steps 

		Q = np.zeros((num_rows, num_columns, num_actions), dtype=float)

	episodes_steps = np.cumsum(episodes_steps)
	return (episodes_steps)/num_runs

if __name__ == '__main__':
	
	num_rows = 7
	num_columns = 10
	alpha = 0.5
	epsilon = 0.1
	num_episodes = 170
	num_runs = 30
	num_actions = 4
	gamma = 1
	wind_dist = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=int)
	start_block = np.asarray([3, 0], dtype=int)
	goal_block = np.asarray([3, 7], dtype=int)
	episodes_steps = sarsa_on_policy(num_rows, num_columns, num_actions, num_episodes, wind_dist, start_block, goal_block, alpha, epsilon, num_runs, gamma)
	episodes = [i for i in range(num_episodes)]
	times = np.zeros(len(episodes_steps)+1)
	times[1:] = episodes_steps
	y = np.zeros(len(episodes)+1)
	y[1:] = episodes

	plt.plot(times, y)
	plt.show()





