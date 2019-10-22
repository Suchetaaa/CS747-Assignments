import numpy as np 
from rt_gen import rt_gen
from pulp import *
import sys

def lp(states, actions, rewards, transitions, gamma, type_mdp):

	lp_problem = pulp.LpProblem("optimal_v", pulp.LpMinimize)

	state_nums = [p for p in xrange(0, states)]
	q_func = np.zeros((states, actions), dtype=float)
	state_variables = pulp.LpVariable.dicts('vpi', state_nums, cat='Continuous')
	lp_problem += pulp.lpSum([state_variables[s] for s in state_nums])
	states_1 = states

	if gamma == 1 and (type_mdp.split() == "episodic".split()):
		lp_problem += state_variables[states-1] == 0
		states_1 = states - 1

	for s in xrange(0, states_1):
		for a in xrange(0, actions):
			lp_problem += state_variables[s] >= pulp.lpSum( (transitions[s, a, s_dash]*( rewards[s, a, s_dash] + gamma * state_variables[s_dash] ) ) for s_dash in xrange(0, states) ) 

	# print lp_problem
	lp_problem.solve()	

	for s in xrange(0,states):
		for a in xrange(0,actions):
			q_func[s, a] = sum( (transitions[s, a, s_dash]*( rewards[s, a, s_dash] + gamma*pulp.value(state_variables[s_dash]))) for s_dash in xrange(0, states) )

	best_policy = np.argmax(q_func, axis=1)
	value_func = np.zeros(states, dtype=np.float128)

	for x in xrange(0,states):
		value_func[x] = pulp.value(state_variables[x])
	# print gamma
	for s in xrange(0,states):
		print "{} {}".format(value_func[s], best_policy[s])
	return best_policy

inp_file = sys.argv[1]
states, actions, rewards, transitions, gamma, type_mdp = rt_gen(inp_file)
lp(states, actions, rewards, transitions, gamma, type_mdp)
