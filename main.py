import numpy as np
import pickle
import copy
import os
from datetime import datetime

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Sensitivity analysis on these params
n_agents = 2
sig = 0.4
inven_factor = 0.05
# maker rebate is the incentive for liquidity provider
maker_rebate = 0.1
# taker fee is the cost for liquidity taker, usually >= maker_rebate
taker_fee = 0.15
GAMMA = 0.5

# Params unchanged
n_iter = 1000000
eps = 1e-5
alpha = 0.05
n_instance = 20
n_tick = 4
n_states = n_tick**2
# n_actions is for one side, then n_actions**2 is for joint ask/bid action
n_actions = 4

tick_space = np.arange(1, n_tick+1)
weights =  0.2*np.maximum(tick_space + taker_fee - 1, 0.0)**2

quote_dist = np.ones((n_actions, n_tick))*0.3/(n_actions-1)
for i in range(n_actions):
    quote_dist[i, i] = 0.7

#rngs for sampling
ss = np.random.SeedSequence(12345)
child_seeds = ss.spawn(n_agents)
rng = [np.random.default_rng(s) for s in child_seeds]


def reward_cal(mo_size, LOB):
    # Compute profits for all agents
    # mo_size: int, LOB: n_agents * quotes
    reward = np.zeros(n_agents)
    order = np.zeros(n_agents)

    if mo_size == 0:
        return order, reward

    # the smallest index to fill market orders
    indx = np.where(np.cumsum(np.sum(LOB, axis=0)) >= mo_size)[0]
    if len(indx) == 0:
        # eat all limit orders
        for agent in range(n_agents):
            reward[agent] = np.dot(LOB[agent], tick_space + maker_rebate)
            order[agent] = np.sum(LOB[agent])
    elif indx[0] > 0:
        for agent in range(n_agents):
            reward[agent] = np.dot(LOB[agent, :indx[0]], tick_space[:indx[0]] + maker_rebate)
            order[agent] = np.sum(LOB[agent, :indx[0]])
        order_left_dist = (mo_size - np.sum(LOB[:, :indx[0]]))*LOB[:, indx[0]]/np.sum(LOB[:, indx[0]])
        order += order_left_dist
        reward += order_left_dist*(tick_space[indx[0]] + maker_rebate)
    elif indx[0] == 0:
        order_dist = mo_size * LOB[:, indx[0]] / np.sum(LOB[:, indx[0]])
        order = order_dist
        reward = order_dist * (tick_space[0] + maker_rebate)

    return order, reward


def act_select(agent, state, eps_threshold):
    if rng[agent].random() > eps_threshold:
        return Q[agent, state].argmax()
    else:
        return rng[agent].integers(0, n_actions**2, 1, dtype=int)


price_ins = np.zeros(n_instance)
mo_ins = np.zeros(n_instance)

Q_hist = []
I_hist = []
mo_hist = []
price_hist = []

for sess in range(n_instance):
    epiQ_hist = []
    epiI_hist = []

    steps_done = 0

    state_hist = np.zeros(1000, dtype=int)
    avg_price = np.zeros(1000)
    total_mo = np.zeros(1000)

    # Initialize the environment and state
    state = rng[0].integers(0, n_states, size=1)
    state_hist[0] = state
    # Counter for variations in heat
    count = 0

    inventory = np.zeros(n_agents)
    Q = np.zeros((n_agents, n_states, n_actions**2))
    # Q = rng[0].random((n_agents, n_states, n_actions**2))

    for i_episode in range(n_iter):
        # For each agent, select and perform an action
        action = np.zeros(n_agents, dtype=int)
        eps_threshold = np.exp(-eps * steps_done)

        for i in range(n_agents):
            action[i] = act_select(i, state, eps_threshold)

        ask_action = (action // n_actions).astype(int)
        bid_action = (action % n_actions).astype(int)

        # interpret the actions as dist of orders
        ask_LOB = np.zeros((n_agents, n_tick))
        bid_LOB = np.zeros((n_agents, n_tick))

        for agent in range(n_agents):
            if inventory[agent] > 500:
                ask_LOB[agent, :] = np.floor(quote_dist[ask_action[agent], :]*20)
                bid_LOB[agent, -1] = 1
            elif inventory[agent] < -500:
                ask_LOB[agent, -1] = 1
                bid_LOB[agent, :] = np.floor(quote_dist[bid_action[agent], :]*20)
            else:
                ask_LOB[agent, :] =  np.floor(quote_dist[ask_action[agent], :]*20)
                bid_LOB[agent, :] =  np.floor(quote_dist[bid_action[agent], :]*20)

        if np.sum(ask_LOB) == 0:
            ask_mo = 0
        else:
            ask_arr_rate = np.exp(-np.dot(np.sum(ask_LOB, axis=0)/np.sum(ask_LOB)/sig, weights))
            ask_mo = rng[0].binomial(n=20*n_agents, p=ask_arr_rate, size=1)[0]

        if np.sum(bid_LOB) == 0:
            bid_mo = 0
        else:
            bid_arr_rate = np.exp(-np.dot(np.sum(bid_LOB, axis=0)/np.sum(bid_LOB)/sig, weights))
            bid_mo = rng[1].binomial(n=20*n_agents, p=bid_arr_rate, size=1)[0]

        bid_filled, bid_reward = reward_cal(bid_mo, bid_LOB)
        ask_filled, ask_reward = reward_cal(ask_mo, ask_LOB)
        inventory = inventory + bid_filled - ask_filled

        # Include inventory risk
        reward_total = bid_reward + ask_reward - inven_factor*(bid_filled - ask_filled)**2

        # Average price, except the inventory
        if bid_filled.min() !=0 and ask_filled.min() !=0:
            avg_price[steps_done%1000] = (np.divide(bid_reward, bid_filled) + np.divide(ask_reward, ask_filled)).mean()

        # Average market order
        total_mo[steps_done%1000] = ask_mo + bid_mo

        # Observe new state by evaluating current ask/bid levels
        # ask_state = np.argmax(np.sum(ask_LOB, axis=0))
        # bid_state = np.argmax(np.sum(bid_LOB, axis=0))

        ask_state = np.round(np.dot(np.sum(ask_LOB, axis=0), tick_space-1)/np.sum(ask_LOB))
        bid_state = np.round(np.dot(np.sum(bid_LOB, axis=0), tick_space-1)/np.sum(bid_LOB))

        next_state = (ask_state*n_actions + bid_state).astype(int)


        old_heat = Q.argmax(2)
        old_Qmax = Q.max()
        for agent in range(n_agents):
            change = reward_total[agent] + GAMMA * Q[agent, next_state].max() - Q[agent, state, action[agent]]
            Q[agent, state, action[agent]] += alpha * change

        new_heat = Q.argmax(2)
        new_Qmax = Q.max()

        if np.sum(np.abs(old_heat - new_heat)) == 0:
            count += 1
        else:
            count = 0

        state = next_state
        state_hist[steps_done%1000] = state
        steps_done += 1

        if i_episode%50000 == 0:
            epiQ_hist.append(copy.deepcopy(Q))
            epiI_hist.append(copy.deepcopy(inventory))

        # if i_episode % 100000 == 0:
        #     print(bcolors.GREEN + 'Instance', sess, 'Count', count, 'Steps done:', steps_done, bcolors.ENDC)
        #     print('Inventory', inventory)
        #     print('Ask LOB', ask_LOB)
        #     print('Bid LOB', bid_LOB)
        #     print('State', state)
        #     print('Ask MO', ask_mo, 'Bid MO', bid_mo)
        #     print(bcolors.PINK + 'Bid Action', bid_action, bcolors.ENDC)
        #     print(bcolors.PINK + 'Ask Action', ask_action, bcolors.ENDC)

    print( 'Instance', sess, 'Count', count, 'Steps done:', steps_done)
    print('Average price:', avg_price.mean())
    print('Average market order:', total_mo.mean())
    if count > 100000:
        print(bcolors.GREEN + 'Terminate condition satisfied with state', np.array(state_hist[-10:]), bcolors.ENDC)


    price_ins[sess] = avg_price.mean()
    mo_ins[sess] = total_mo.mean()

    Q_hist.append(epiQ_hist)
    I_hist.append(epiI_hist)
    mo_hist.append(copy.deepcopy(total_mo))
    price_hist.append(copy.deepcopy(avg_price))

print(price_ins)
print(mo_ins)


sub_folder = '{}_{:.2f}_{}.{}'.format('gam50', maker_rebate,
                                      datetime.now().strftime('%M'), datetime.now().strftime('%S'))

log_dir = './logs/{}'.format(sub_folder)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save params configuration
with open('{}/params.txt'.format(log_dir), 'w') as fp:
    fp.write('Params setting \n')
    fp.write('Discount factor {} \n'.format(GAMMA))
    fp.write('No. of agents {} \n'.format(n_agents))
    fp.write('Volatility {} \n'.format(sig))
    fp.write('Inventory aversion {} \n'.format(inven_factor))
    fp.write('Maker rebate {} \n'.format(maker_rebate))
    fp.write('Taker fee {} \n'.format(taker_fee))


with open('{}/Q.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(Q_hist, fp)

with open('{}/inven.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(I_hist, fp)

with open('{}/mo.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(mo_hist, fp)

with open('{}/price.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(price_hist, fp)
