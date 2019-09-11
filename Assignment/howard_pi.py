import numpy as np 
import sys 
from pulp import *
from rt_gen import rt_gen

def get_qfunc(policy, states, actions, rewards, transitions, gamma):
	q_problem = pulp.LpProblem("finding_Q")
	state_nums = [p for p in xrange(0, states)]
	state_variables = pulp.LpVariable.dicts('qpi', state_nums, cat='Continuous')

	for s in xrange(0, states):
		q_problem += state_variables[s] == pulp.lpSum( (transitions[s, policy[s], s_dash]*( rewards[s, policy[s], s_dash] + gamma * state_variables[s_dash] ) ) for s_dash in xrange(0, states) ) 

	print q_problem
	q_problem.solve()

	q_func = np.zeros(states, dtype=float)
	for x in xrange(0, states):
		q_func[x] = pulp.value(state_variables[x])
	return q_func

def update_policy(curr_policy, states, actions, q_func):
	new_policy = np.zeros(zeros, dtype=int)
	for s in xrange(0, states):
		for a in xrange(0, actions):
			
			
inp_file = sys.argv[1]
states, actions, rewards, transitions, gamma, type_mdp = rt_gen(inp_file)

if type_mdp == 'episodic':
	gamma = 1

curr_policy = np.zeros(states, dtype=int)


