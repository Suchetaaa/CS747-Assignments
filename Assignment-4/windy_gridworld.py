import numpy as np
import matplotlib.pyplot as plt 
import random
import sys

def nextstate_reward(num_rows, num_columns, wind_dist, start_block, goal_block, present_state, action, stochastic, seed):
	reward = -1
	if stochastic == 0:
		wind_strength = wind_dist[int(present_state[1])]
	else: 
		wind_strength = wind_dist[int(present_state[1])] + random.randint(-1, 1)

	next_state = np.asarray([0, 0], dtype=int)

	# up
	if action == 0:
		next_state[0] = min(max(present_state[0] - 1 - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1], 0), num_columns-1)

	# down
	elif action == 1:
		next_state[0] = min(max(present_state[0] + 1 - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1], 0), num_columns-1)

	# right
	elif action == 2:
		next_state[0] = min(max(present_state[0] - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1] + 1, 0), num_columns-1)
	
	# left
	elif action == 3:
		next_state[0] = min(max(present_state[0] - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1] - 1, 0), num_columns-1)

	# north-east
	elif action == 4: 
		next_state[0] = min(max(present_state[0] - 1 - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1] + 1, 0), num_columns-1)

	# north-west
	elif action == 5:
		next_state[0] = min(max(present_state[0] - 1 - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1] - 1, 0), num_columns-1)

	# south-east
	elif action == 6:
		next_state[0] = min(max(present_state[0] + 1 - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1] + 1, 0), num_columns-1)

	# south-west
	elif action == 7:
		next_state[0] = min(max(present_state[0] + 1 - wind_strength, 0), num_rows-1)
		next_state[1] = min(max(present_state[1] - 1, 0), num_columns-1)

	if next_state[0] == goal_block[0] and next_state[1] == goal_block[1]:
		reward = 0

	return next_state, reward

def epsilon_greedy(present_state, epsilon, num_actions, Q, seed):
	# np.random.seed(seed)
	random1 = np.random.rand(1, 1)
	action = 0
	if random1 < epsilon: 
		action = random.randint(0, num_actions-1)
	else:
		action = np.amax(np.argmax(Q[present_state[0]][present_state[1]]))

	return action

def sarsa_on_policy(num_rows, num_columns, num_actions, num_episodes, wind_dist, start_block, goal_block, alpha, epsilon, num_runs, gamma, stochastic):
	Q = np.zeros((num_rows, num_columns, num_actions), dtype=float)
	episodes_steps = np.zeros(num_episodes, dtype=float)

	for n in range(0, num_runs):
		# print (n)
		seed = 2*n
		# np.random.seed(seed)
		for x in range(0,num_episodes):
			present_state = start_block
			action = epsilon_greedy(present_state, epsilon, num_actions, Q, seed)
			time_steps = 0

			while(1):
				state_dash, reward = nextstate_reward(num_rows, num_columns, wind_dist, start_block, goal_block, present_state, action, stochastic, seed)
				action_dash = epsilon_greedy(state_dash, epsilon, num_actions, Q, seed)
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
	alpha = float(sys.argv[1])
	epsilon = float(sys.argv[2])
	num_episodes = int(sys.argv[3])
	num_runs = 10
	num_actions = 4
	gamma = 1
	stochastic = 0
	wind_dist = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=int)
	start_block = np.asarray([3, 0], dtype=int)
	goal_block = np.asarray([3, 7], dtype=int)
	episodes_steps1 = sarsa_on_policy(num_rows, num_columns, num_actions, num_episodes, wind_dist, start_block, goal_block, alpha, epsilon, num_runs, gamma, stochastic)
	episodes1 = [i for i in range(num_episodes)]
	times1 = np.zeros(len(episodes_steps1)+1)
	times1[1:] = episodes_steps1
	y1 = np.zeros(len(episodes1)+1)
	y1[1:] = episodes1

	num_actions = 8
	episodes_steps2 = sarsa_on_policy(num_rows, num_columns, num_actions, num_episodes, wind_dist, start_block, goal_block, alpha, epsilon, num_runs, gamma, stochastic)
	episodes2 = [i for i in range(num_episodes)]
	times2 = np.zeros(len(episodes_steps2)+1)
	times2[1:] = episodes_steps2
	y2 = np.zeros(len(episodes2)+1)
	y2[1:] = episodes2

	stochastic = 1
	episodes_steps3 = sarsa_on_policy(num_rows, num_columns, num_actions, num_episodes, wind_dist, start_block, goal_block, alpha, epsilon, num_runs, gamma, stochastic)
	episodes3 = [i for i in range(num_episodes)]
	times3 = np.zeros(len(episodes_steps3)+1)
	times3[1:] = episodes_steps3
	y3 = np.zeros(len(episodes3)+1)
	y3[1:] = episodes3
	
	# print (y1)
	# print (y2)
	# print (y3)

	plt.figure()
	plt.plot(times1, y1)
	plt.title("Basic Windy Gridworld, learning rate={}, epsilon={}".format(alpha, epsilon))
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.show()

	plt.figure()
	plt.plot(times2, y2)
	plt.title("Kings Moves Windy Gridworld, learning rate={}, epsilon={}".format(alpha, epsilon))
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.show()

	plt.figure()
	plt.plot(times3, y3)
	plt.title("Kings Moves & Stochastic Wind, learning rate={}, epsilon={}".format(alpha, epsilon))
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.show()

	plt.figure()
	plt.plot(times1, y1, label="Basic")
	plt.plot(times2, y2, label="With king's moves")
	plt.plot(times3, y3, label="King's moves & stochastic wind")
	plt.title("Windy Gridworld, learning rate={}, epsilon={}".format(alpha, epsilon))
	plt.legend(loc='lower right')
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.show()
