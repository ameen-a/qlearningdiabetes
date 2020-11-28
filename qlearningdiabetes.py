%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt

"""

Bugs:
- the 2nd action index of the Q-value is rarely changing from 0. As a result, it's ever chosen.
- figure out the data flow of the Q-value in relation to the action index. Make sure it's being
 properly translated over.

# every X hours there is a meal administered which increases the amount 
# there is an upper and lower limit after which the user experiences either hyper or hypoglycemia
# 5 second incremenets and 24 hours in day; 288 incrememnts
# start at 8AM, end at 8AM following day
# meal at 9AM, 1PM, 7PM corresponding to iteration 

TODO:
- performance evaluation function, # create new array which stores number of 
# the number of correct ts
# def eval_performance(t, perf, blood_glucose, min_bg, max_bg):
#     if min_bg <= blood_glucose <= max_bg:

"""


# reward agent if bg level is within range 
def calculate_reward(blood_glucose, min_bg, max_bg):
    if min_bg <= blood_glucose <= max_bg:
        reward = 5
        return reward
    else:
        return -1000 
    

# choose action based on epsilon-greedy
def choose_action(step, epsilon):
    if np.random.random() < epsilon:
        
        return np.argmax(q_values[step])
    else: 
        return np.random.randint(1)
    

# training parameters
epsilon = 0.90
discount_factor = 0.9
learning_rate = 0.9
episodes = 300

# BG values
initial_blood_glucose = 500
max_bg = 900
min_bg = 350
bg_dose = 100
bg_decay = 3
    
# time and iterations
t_max = 1440
timestep = 5
num_timesteps = int(t_max / timestep)


# q-values; 288x2, each time incremement has 2 layers for each action
q_values = np.zeros((num_timesteps, 2))


for episode in range(episodes):
    
    # initialise starting values
    blood_glucose = initial_blood_glucose
    t = 0
    reward = 0
    
    # value storage
    t_list = []
    bg_list = []
    action_list = []
    t_list.append(t)
    bg_list.append(blood_glucose)
    
    
    # begin simulation environment containing t and BG
    for step in range(num_timesteps):
        
        # apply bg decay
        blood_glucose -= bg_decay 

        
        # administer glucose for meal times
        if t == 12 or t == 60 or t == 132:
            blood_glucose += 200
        
        # choose action according to epsilon-greedy
        action_index = choose_action(step, epsilon)
        # take insulin
        if action_index == 1:
            print("episode {}: adminstered insulin at {} because level was {}".format(episode, step, blood_glucose))
            blood_glucose += bg_dose
            
            
        # allocate reward
        reward += calculate_reward(blood_glucose, min_bg, max_bg)
        
        # calculate TD
        old_q_value = q_values[step, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[step, action_index])) - old_q_value
        
        # update q-value for previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[step, action_index] = new_q_value
        
        # advance timestep and store t and bg
        t += 5
        t_list.append(t)
        bg_list.append(blood_glucose)
    
        

plt.plot(t_list, bg_list)
plt.show()
