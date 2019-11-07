"""
This script is about the monte carlo method of the Monty Hall problem, Creating by Peter Lee On October 22, 2018s.
"""
import random


def Monty_Hall_simulation(strategy, simulation_number):
    wins_count = 0
    # Simulated 100000 times based on monte carlo
    for i in range(simulation_number):
        # 0 means car and 1,2 means goats
        choice_list_init = [0, 1, 2]
        first_choice = random.choice(choice_list_init)
        # According to the precondition, the compere helps us eliminate an error option, so the sample must include the
        # car and a goat.
        if first_choice == 0:
            # the first time we choice the car
            sample_space = [0, random.choice([1, 2])]
        else:
            sample_space = [0, first_choice]
        # Counting the simulation results on the condition of stick the first choice or change the choice.
        if strategy == 'stick':
            result = first_choice
        if strategy == 'change':
            sample_space.pop(sample_space.index(first_choice))
            result = sample_space[0]
        if result == 0:
            wins_count += 1
    win_probability = round(wins_count/simulation_number, 6)
    print("The probability of win in {0} strategy is {1}: " .format(strategy, win_probability))


Monty_Hall_simulation('change', 100000)
Monty_Hall_simulation('stick', 100000)