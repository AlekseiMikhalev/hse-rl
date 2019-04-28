
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    q = 0
    
    for state_prime, p in mdp.get_next_states(state, action).items():
         q += p * (mdp.get_reward(state, action, state_prime) + gamma * state_values[state_prime])

    return q
