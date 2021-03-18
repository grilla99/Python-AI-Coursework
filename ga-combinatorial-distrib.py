# -*- coding: utf-8 -*-
# Use standard python package only.
import random, string
import math
import numpy as np
import matplotlib as plt
import City

from math import cos, pi, sin,sqrt

# MINIMUM GLOBAL VARIABLES TO BE USED
POPULATION_SIZE = 100   # Change POPULATION_SIZE to obtain better fitness.

GENERATIONS = 100  # Change GENERATIONS to obtain better fitness.
SOLUTION_FOUND = False

CROSSOVER_RATE = 0.8  # Change CROSSOVER_RATE  to obtain better fitness.
MUTATION_RATE = 0.2  # Change MUTATION_RATE to obtain better fitness.

LOWER_BOUND = -10
UPPER_BOUND = 10
FITNESS_CHOICE = 4
NUMBER_TO_REACH = 60

NO_OF_GENES = 8
NO_OF_PARENTS = 8
NO_OF_CITIES = 12

KNAPSACK = {}
KNAPSACK_WEIGHT_THRESHOLD = 35


FITNESS_DICTIONARY = {
    1: "Sum 1s (Minimisation)",
    2: "Sum 1s (Maximisation)",
    3: "String matching",
    4: "Reaching a number",
    5: "Knapsack problem",
    6: "Travelling Salesman"
}


def generate_population(size):
    # Sum 1s (Min and Max)
    if FITNESS_CHOICE == 1 or FITNESS_CHOICE == 2:
        population = [[random.randint(0, 1) for _ in range(NO_OF_GENES)] for _ in range(POPULATION_SIZE)]
    # String Matching
    elif FITNESS_CHOICE == 3:
        # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789
        # Reduced to just ascii letters as wasn't capable of learning the full char set listed above
        # terms = string.ascii_letters + string.punctuation + string.digits
        terms = string.ascii_lowercase
        population = [[''.join(random.choice(terms) for _ in range(NO_OF_GENES))] for _ in range(POPULATION_SIZE)]
    # Reaching a number
    elif FITNESS_CHOICE == 4:
        population = [[random.randint(LOWER_BOUND, UPPER_BOUND) for _ in range(NO_OF_GENES)] for _ in
                      range(POPULATION_SIZE)]
    # Travelling Salesman
    elif FITNESS_CHOICE == 6:
        # Assuming a 'board size' of 50. Change this if necessary
        population = [[City.City(x, random.randrange(0, 50), random.randrange(0, 50)) for x in range(NO_OF_CITIES)] for
                       _ in range(POPULATION_SIZE)]

    return population


def compute_fitness(individual):
    fitness_function = {
        1: sum_ones,
        2: sum_ones,
        3: match_string,
        4: reach_number,
        5: knapsack_problem,
        6: travelling_salesman
    }

    fitness_func = fitness_function.get(FITNESS_CHOICE)

    return fitness_func(individual)


def sum_ones(individual):
    fitness = 0
    if FITNESS_CHOICE == 1:
        digit_to_match = 0
    else:
        digit_to_match = 1

    for x in individual:
        if x == digit_to_match:
            fitness += 1

    return fitness


#TODO: Needs some work as most strings return a fitness of 0
def match_string(individual):
    fitness = 0
    string_to_match = 'eutopia'

    for x in range(len(individual)):
        if individual[0][x] == string_to_match[x]:
            fitness += 1

    return fitness


def reach_number(individual):
    fitness = 0

    number_to_reach = NUMBER_TO_REACH

    fitness = number_to_reach - sum(individual)

    try:
        fitness = (1 / abs(fitness) * 100)
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


def travelling_salesman(individual):
    fitness = 0

    for x in range(len(individual)):
        try:
            city1 = individual[x]
            city2 = individual[x+1]
        except IndexError:
            city2 = individual[0]

        fitness += city1.distance(city2)

    # 100 produced very low numbers
    fitness = abs(1/fitness) * 1000
    return fitness



# https://www.youtube.com/watch?v=MacVqujSXWE
# https://medium.com/koderunners/genetic-algorithm-part-3-knapsack-problem-b59035ddd1d6
def knapsack_problem(individual):
    sack_weight = 0
    sack_value = 0

    for x in range(len(individual)):
        if individual[x] == 1:
            sack_weight += KNAPSACK[x][0]
            sack_value += KNAPSACK[x][1]

    # If the bag exceeds the threshold weight, then isn't an acceptable solution
    # So the function will return 0 to signify thiss
    if sack_weight > KNAPSACK_WEIGHT_THRESHOLD:
        return 0
    else:
        return sack_value


def selection(population, fitness, no_of_parents):

    parents = np.empty((no_of_parents, NO_OF_GENES))

    # Declaration required to avoid referenced before assignment error
    positive_fitness = fitness.copy()

    # Roulette Wheel Selection Operator
    if (sum(i < 0 for i in fitness) != 0):
        positive_fitness = [fitness[x] + abs(min(fitness)) + 1 for x in range(len(fitness))]

    total_fitness = sum(positive_fitness)

    try:
        relative_fitness = [(n / total_fitness) for n in positive_fitness]
        roulette_indices = np.random.choice(range(0, len(population)), size=NO_OF_PARENTS, replace=False,
                                            p=relative_fitness)
    except ZeroDivisionError:
        print("Population fitness of 0")
        return False

    parents = [population[x] for x in roulette_indices]

    return parents
    

def crossover(parents, num_of_offspring):
    offspring = np.empty((num_of_offspring, NO_OF_GENES))

    for i in range(num_of_offspring):
        parent1_index = i % NO_OF_PARENTS
        parent2_index = (i+1) % NO_OF_PARENTS

        if random.random() < CROSSOVER_RATE:
            offspring[i] = list(parents[parent1_index][0:4]) + list(parents[parent2_index][4:9])
        else:
            offspring[i] = list(parents[parent1_index])

    return offspring


def mutation(offspring):

    # Scramble mutation for BINARY
    if random.random() < MUTATION_RATE:
        no_of_mutations = random.randint(0, NO_OF_GENES / 2)
        affected_gene = random.randint(0, NO_OF_GENES / 2)

        if no_of_mutations >= 1 and (FITNESS_CHOICE == 1 or FITNESS_CHOICE == 2):
            for x in range(no_of_mutations):
                # Swaps the 0 to 1 and 1 to 0 for Binary Problems
                offspring[affected_gene + x] = 1 - offspring[affected_gene + x]
        elif no_of_mutations >= 1 and FITNESS_CHOICE == 3:
            # Swaps character for another alphanumeric character
            for x in range(no_of_mutations):
                terms = string.ascii_letters + string.punctuation + string.digits
                offspring[affected_gene] = random.choice(terms)
        elif no_of_mutations >= 1 and FITNESS_CHOICE == 4:
            for x in range(no_of_mutations):
                # Swaps to a random integer in allowed range (-10 to 10 for most runs)
                offspring[affected_gene + x] = random.randint(LOWER_BOUND, UPPER_BOUND)

    return offspring


def check_solution(population):
    # Sum 1s (Minimisation)
    if FITNESS_CHOICE == 1:
        ideal_solution = [0 for _ in range(NO_OF_GENES)]
    # Sum 1s (Maximisation)
    elif FITNESS_CHOICE == 2:
        ideal_solution = [1 for _ in range(NO_OF_GENES)]
    # Matching a string
    elif FITNESS_CHOICE == 3:
        ideal_solution = "eutopia"
    # Reaching a number
    elif FITNESS_CHOICE == 4:
        ideal_solution = NUMBER_TO_REACH
        for x in population:
            if sum(x) == ideal_solution:
                return True
    # Solution checking implemented elsewhere for FITNESS_CHOICE 5 and 6
    else:
        return False

    for x in population:
        if x == ideal_solution:
            return True

    return False
    

# USE THIS MAIN FUNCTION TO COMPLETE YOUR CODE - MAKE SURE IT WILL RUN FROM COMOND LINE   
def main(): 
    global POPULATION_SIZE 
    global GENERATIONS
    global SOLUTION_FOUND

    print("Parameters for run: \n")
    print("Generations: \n", GENERATIONS)
    print("Number of parents: \n", NO_OF_PARENTS)
    print("Number of genes: \n", NO_OF_GENES)
    print("Mutation rate: \n", MUTATION_RATE)
    print("Crossover rate: \n", CROSSOVER_RATE)

    gen_count = 1

    population = generate_population(POPULATION_SIZE)

    fitness = [compute_fitness(x) for x in population]

    if check_solution(population):
        print("Best individual found in", gen_count, " generations")
        SOLUTION_FOUND = True
    else:
        gen_count += 1

    while gen_count <= GENERATIONS and SOLUTION_FOUND != True:
        next_gen = []

        parents = selection(population, fitness, NO_OF_PARENTS)
        offspring = crossover(parents, POPULATION_SIZE-NO_OF_PARENTS)
        offspring = [mutation(x) for x in offspring]

        next_gen += offspring
        population = next_gen

        fitness = [compute_fitness(x) for x in population]
        # Index of fittest individual
        fitness_index = fitness.index(max(fitness))

        best_individual = population[fitness_index]

        print("Generation: ", gen_count, " Max fitness: ", max(fitness), " Best individual: ", best_individual)

        gen_count += 1

    if FITNESS_CHOICE == 5:
        for x in range(NO_OF_GENES):
            KNAPSACK[x] = (random.randint(1, 15), random.randint(0, 600))

    if FITNESS_CHOICE == 5:
        print('Knapsack List is as follows: ')
        for x in KNAPSACK:
            print("Item No: ", x, "Weight: ", KNAPSACK[x][0], "Value: ", KNAPSACK[x][1])

    print('complete code for a combinitorial optimization problem:')
    while (True):  # TODO: write your termination condition here or within the loop 
        #TODO: write your generation propagation code here 


        #TODO: present innovative graphical illustration like plots and presentation of genetic algorithm results 
        #This is free (as you like) innovative part of the assessment.
        break # Remove this line
 

if __name__ == '__main__': 
    main() 
    
    
    
