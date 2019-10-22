import numpy as np 
import sys 
from pulp import *
from rt_gen import rt_gen

def get_qfunc(policy, states, actions, rewards, transitions, gamma, type_mdp):
	q_problem = pulp.LpProblem("finding_Q")
	state_nums = [p for p in xrange(0, states)]
	state_variables = pulp.LpVariable.dicts('qpi', state_nums, cat='Continuous')
	states_1 = states

	if gamma == 1 and type_mdp.split() == "episodic".split():
		q_problem += state_variables[states-1] == 0
		states_1 = states - 1

	for s in xrange(0, states_1):
		q_problem += state_variables[s] == pulp.lpSum( (transitions[s, policy[s], s_dash]*( rewards[s, policy[s], s_dash] + gamma * state_variables[s_dash] ) ) for s_dash in xrange(0, states) ) 

	# print q_problem
	q_problem.solve()

	q_func = np.zeros(states, dtype=np.float128)
	for x in xrange(0, states):
		q_func[x] = pulp.value(state_variables[x])
	return q_func

def update_policy(curr_policy, states, actions, q_func, transitions, rewards, gamma):
	new_policy = curr_policy.copy()
	for s in xrange(0, states):
		for a in xrange(0, actions):
			q_action = sum((transitions[s, a, s_dash]*( rewards[s, a, s_dash] + gamma * q_func[s_dash] ) ) for s_dash in xrange(0, states) )
			if q_action > q_func[s] + 1e-6:
				new_policy[s] = a
				# break
	return new_policy

def howard_pi(states, actions, rewards, transitions, gamma, type_mdp):

	curr_policy = np.zeros(states, dtype=int)

	while True:
		curr_q_func = get_qfunc(curr_policy, states, actions, rewards, transitions, gamma, type_mdp)
		new_policy = update_policy(curr_policy, states, actions, curr_q_func, transitions, rewards, gamma)
		if (new_policy == curr_policy).all():
			curr_q_func = get_qfunc(new_policy, states, actions, rewards, transitions, gamma, type_mdp)
			break
		curr_policy = new_policy

	# print curr_q_func
	# print gamma
	# print new_policy
	for s in xrange(0,states):
		print "{} {}".format(curr_q_func[s], new_policy[s])
	return curr_policy

inp_file = sys.argv[1]
states, actions, rewards, transitions, gamma, type_mdp = rt_gen(inp_file)
howard_pi(states, actions, rewards, transitions, gamma, type_mdp) 
	