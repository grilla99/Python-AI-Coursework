# -*- coding: utf-8 -*-
"""
Complete this code for continuous optimization  problem

Please remove author name if generated by machine automatically
Keep you code anonymous

"""

# Use standard python package only.
import random
import math
import numpy as np
import matplotlib as plt


# MINIMUM GLOBAL VARIABLES TO BE USED
POPULATION_SIZE = 50 # Change POPULATION_SIZE to obtain better fitness.

GENERATIONS = 100 # Change GENERATIONS to obtain better fitness.
SOLUTION_FOUND = False

CORSSOVER_RATE = 0.8 # Change CORSSOVER_RATE  to obtain better fitness.
MUTATION_RATE = 0.2 # Change MUTATION_RATE to obtain better fitness.

LOWER_BOUND = -10
UPPER_BOUND = 10
FITNESS_CHOICE = 1

NO_OF_PARENTS = 2
NO_OF_GENES = 8


def generate_population(size, lower_bound, upper_bound):
    population = [generate_individual(NO_OF_GENES, lower_bound, upper_bound)
                  for _ in range(size)]
    # Returns an array of individuals, each composed of 1 or more ints
    return population


def generate_individual(no_of_genes, lower_bound, upper_bound):
    # Individual is a list of random ints in a range
    return [random.randrange(lower_bound, upper_bound) for _ in range(no_of_genes)]


# Function used to calculate the fitness of an individual dependent on the problem to be solved
def compute_fitness(individual):
    fitness_function = {
        1: sum_square
    }

    fitness_func = fitness_function.get(FITNESS_CHOICE)

    return fitness_func(individual)


def sum_square(individual):
    fitness = sum([(x+1) * individual[x]**2 for x in range(NO_OF_GENES)])
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness



def selection(population, fitness):

    individual = []  # Update this line if you need to

    # Declaration required to avoid referenced before assignment error
    positive_fitness = fitness.copy()

    ## ROULETTE WHEEL SELECTION OPERATOR ##
    if (sum(i < 0 for i in fitness) != 0):
        positive_fitness = [fitness[x] + abs(min(fitness)) + 1 for x in range(len(fitness))]

    total_fitness = sum(positive_fitness)

    try:
        # Computes the relative likelihoods for each parent to be chosen
        relative_fitness = [(n / total_fitness) for n in positive_fitness]
        #Uses numpy to create a random sample using the relative fitness
        roulette_indices = np.random.choice(range(0, POPULATION_SIZE), size=NO_OF_PARENTS, replace=False, p=relative_fitness)

    except ZeroDivisionError:
        print("Population fitness of 0")
        roulette_indices = np.random.choice(range(0, POPULATION_SIZE), size=NO_OF_PARENTS, replace=False)

    parents = [population[x] for x in roulette_indices]
    print(parents)

    return individual


def crossover(first_parent, second_parent):

    individual = [] # Update this line if you need to
    #TODO : Write your own code to for choice  of your crossover operator - you can use if condition to write more tan one ime of crossover operator

    return individual

def mutation(individual):

    #TODO : Write your own code to for choice  of your mutation operator - you can use if condition to write more tan one ime of crossover operator


    return individual

#TODO : You can increase number of function to be used to improve your GA code




def next_generation(previous_population):
    #TODO : Write your own code to generate next


    print(' ') # Print appropriate generation information here.
    return next_generation


# USE THIS MAIN FUNCTION TO COMPLETE YOUR CODE - MAKE SURE IT WILL RUN FROM COMOND LINE
def main():
    global POPULATION_SIZE
    global GENERATIONS
    global SOLUTION_FOUND

    lower_bound = [] #Update this
    upper_bound = [] #Update this

    population = generate_population(POPULATION_SIZE, LOWER_BOUND, UPPER_BOUND)

    for x in range(len(population)):
        print(compute_fitness(population[x]))

    fitness = [compute_fitness(x) for x in population]

    print(selection(population, fitness))


    print('complete code for a continuous optimization problem:')
    while (True):  # TODO: write your termination condition here or within the loop
        #TODO: write your generation propagation code here


        #TODO: present innovative graphical illustration like plots and presentation of genetic algorithm results
        #This is free (as you like) innovative part of the assessment.
        break;


if __name__ == '__main__':
    main()



