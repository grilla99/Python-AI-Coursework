# -*- coding: utf-8 -*-
# Use standard python package only.
import random
import math
import numpy as np
import matplotlib as plt


from math import cos, pi, sin,sqrt

# MINIMUM GLOBAL VARIABLES TO BE USED
POPULATION_SIZE = 50  # Change POPULATION_SIZE to obtain better fitness.

GENERATIONS = 100 # Change GENERATIONS to obtain better fitness.
SOLUTION_FOUND = False

CROSSOVER_RATE = 0.8 # Change CROSSOVER_RATE  to obtain better fitness.
MUTATION_RATE = 0.6 # Change MUTATION_RATE to obtain better fitness.

LOWER_BOUND = -10
UPPER_BOUND = 10
FITNESS_CHOICE = 6

RUN_CHOICE = 2

NO_OF_PARENTS = 8
NO_OF_GENES = 8
NO_OF_OFFSPRING = 1


FITNESS_DICTIONARY = {
    1: "Sum Squares",
    2: "Rastrigin",
    3: "Dixon Price",
    4: "Rosenbrock",
    5: "Schwefel",
    6: "Trid"
}


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
        1: sum_square,
        2: rastrigin,
        3: dixon_price,
        4: rosenbrock,
        5: schwefel,
        6: trid
    }

    fitness_func = fitness_function.get(FITNESS_CHOICE)

    return fitness_func(individual)


def sum_square(individual):
    fitness = sum([(x+1) * individual[x]**2 for x in range(NO_OF_GENES)])
    # Case where fitness will be 0 and thus need to account for the error message
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


def rosenbrock(individual):
    fitness = sum((100 * (individual[x+1] - individual[x]**2)**2 + ((individual[x] - 1) ** 2) for x in individual))
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


def rastrigin(individual):
    fitness = 10 * len(individual) + sum([((x - 1)**2 - 10 * cos(2 * pi * (x-1))) for x in individual])
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


def dixon_price(individual):
    fitness = (individual[0]-1)** 2 + sum((x + 1) * (2 * individual[x]**2 - individual[x-1])**2 for x in range(1, NO_OF_GENES))
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

#TODO: Change the INPUT RANGE to be -500 - 500 for this one
def schwefel(individual):
    # Schwefel function defined as 418.9829 * dimensions - .... , here we assume 1 dimension
    fitness = 418 - sum(individual[x] * sin(sqrt(abs(individual[x]))) for x in range(1, NO_OF_GENES))
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


def trid(individual):
    term1 = sum((individual[x]-1)** 2 for x in range(NO_OF_GENES))
    term2 = sum((individual[x] * individual[x-1]) for x in range(1, NO_OF_GENES))

    fitness = term1 - term2
    # Has no local minimum only the global. Defined as -d(d+4)(d-1)/6
    ideal_fitness = -NO_OF_GENES*(NO_OF_GENES+4)*(NO_OF_GENES-1) / 6

    try:
        fitness = abs(1/fitness-ideal_fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


def selection(population, fitness,no_of_parents):

    parents = np.empty((no_of_parents, NO_OF_GENES))

    # Declaration required to avoid referenced before assignment error
    positive_fitness = fitness.copy()

    # Roulette Wheel Selection Operator
    if sum(i < 0 for i in fitness) != 0:
        positive_fitness = [fitness[x] + abs(min(fitness)) + 1 for x in range(len(fitness))]

    total_fitness = sum(positive_fitness)

    try:
        # Computes the relative likelihoods for each parent to be chosen
        relative_fitness = [(n / total_fitness) for n in positive_fitness]
        # Uses numpy to create a random sample using the relative fitness
        roulette_indices = np.random.choice(range(0, POPULATION_SIZE), size=NO_OF_PARENTS, replace=False,
                                            p=relative_fitness)
    except ZeroDivisionError:
        print("Population fitness of 0")
        return False

    parents = [population[x] for x in roulette_indices]

    return parents


def crossover(parents, num_of_offspring):

    offspring = np.empty((num_of_offspring, NO_OF_GENES))

    # Single Point Crossover
    for i in range(num_of_offspring):
        parent1_index = i % NO_OF_PARENTS
        parent2_index = (i+1) % NO_OF_PARENTS

        if random.random() < CROSSOVER_RATE:
            offspring[i] = list(parents[parent1_index][0:4]) + list(parents[parent2_index][4:9])
        else:
            offspring[i] = list(parents[parent1_index])

    return offspring


def mutation(offspring):
    # Random Resetting Mutation
    no_of_mutations = 1
    if RUN_CHOICE == 1:
        for x in range(len(offspring-1)):
            if random.random() < MUTATION_RATE:
                random_index = random.sample(range(NO_OF_GENES), no_of_mutations)
                random_value = random.sample(range(LOWER_BOUND, UPPER_BOUND), no_of_mutations)
                for i in range(len(random_index)):
                    offspring[x][random_index[i]] = random_value[0]

    #Inversion Mutation, Flips - to + and + to -
    elif RUN_CHOICE == 2:
        for x in range(len(offspring-1)):
            if random.random() < MUTATION_RATE:
                random_index = random.sample(range(NO_OF_GENES), no_of_mutations)
                for i in range(len(random_index)):
                    if offspring[x][random_index[i]] > 0:
                        offspring[x][random_index[i]] = offspring[x][random_index[i]] * -1
                    elif offspring[x][random_index[i]] < 0:
                        offspring[x][random_index[i]] = offspring[x][random_index[i]] * -1

    return offspring


def find_best_input(population, fitness):
    best_fitness = max(fitness)
    individual_index = fitness.index(best_fitness)

    return population[individual_index]


def next_generation(generation, fitness, parents, offspring):
    best_fitness = max(fitness)
    print("\n Generation ", generation, ": ",
          "\n Parents Selected: \n", parents,
          "\n Offspring", offspring,
          "\nFitness", fitness,
          "\nBest fitness after generation", best_fitness)


def check_solution(population):
    ## Sum of Squares, Rastrigin, Dixon Price
    if FITNESS_CHOICE == 1 or FITNESS_CHOICE == 2 or FITNESS_CHOICE == 3:
        ideal_individual = [0 for _ in range(NO_OF_GENES)]
    ## Rosenbrock
    elif FITNESS_CHOICE == 4:
        ideal_individual = [1 for _ in range(NO_OF_GENES)]
    ## Schwefel
    elif FITNESS_CHOICE == 5:
        ideal_individual = [420.968 for _ in range(NO_OF_GENES)]
    ## Trid
    elif FITNESS_CHOICE == 6:
        ideal_individual = [ x*(NO_OF_GENES + 1 - x) for x in range(1, NO_OF_GENES+1)]

    for x in population:
        if len([i for i, j in zip(x, ideal_individual) if i == j]) == NO_OF_GENES:
            print("Ideal Individual Found")
            return True

    return False

def run(RUN_NUM, **kwargs):
    lower_bound = LOWER_BOUND
    upper_bound = UPPER_BOUND
    global SOLUTION_FOUND
    global POPULATION_SIZE
    global GENERATIONS
    global NO_OF_PARENTS
    global NO_OF_GENES
    global MUTATION_RATE
    global CROSSOVER_RATE
    global INPUT

    ## Takes optional argument of variable to change, defaults to None if left blank
    var_to_change = kwargs.get('var_to_change', None)
    value = kwargs.get('value', None)

    if var_to_change == "MUTATION":
        MUTATION_RATE = value
    elif var_to_change == "CROSSOVER":
        CROSSOVER_RATE = value
    elif var_to_change == "POPULATION":
        POPULATION_SIZE = value
    elif var_to_change == "NO_OF_PARENTS":
        NO_OF_PARENTS = value

    print("Parameters for run: \n")
    print("Generations: \n", GENERATIONS)
    print("Population: \n", POPULATION_SIZE)
    print("Number of parents: \n", NO_OF_PARENTS)
    print("Mutation rate: \n", MUTATION_RATE)
    print("Crossover rate: \n", CROSSOVER_RATE)

    results_log = {}
    gen_count = 1

    if RUN_NUM == 0:
        population = generate_population(POPULATION_SIZE, lower_bound, upper_bound)
        saved_pop = population.copy()
    elif RUN_NUM != 0 and (var_to_change == "POPULATION"):
        population = generate_population(POPULATION_SIZE, lower_bound, upper_bound)

    fitness = [compute_fitness(x) for x in population]
    results_log[gen_count] = max(fitness)

    while gen_count <= GENERATIONS and SOLUTION_FOUND != True:

        parents = selection(population, fitness, NO_OF_PARENTS)

        offspring = crossover(parents, POPULATION_SIZE - NO_OF_PARENTS)

        offspring = mutation(offspring)

        population = list(parents) + list(offspring)

        fitness = [compute_fitness(x) for x in population]

        if check_solution(population):
            SOLUTION_FOUND = True
        else:
            gen_count += 1

    print("Best Individual", find_best_input(population, fitness))
    print("Outcome of best individual: ", max(fitness))

    if SOLUTION_FOUND:
        results_log.pop(gen_count)
        SOLUTION_FOUND = False

    return results_log


def main():
    global POPULATION_SIZE
    global GENERATIONS
    global SOLUTION_FOUND

    run(0)



         # print('complete code for a continuous optimization problem:')
    while (True):  # TODO: write your termination condition here or within the loop
        #TODO: write your generation propagation code here


        #TODO: present innovative graphical illustration like plots and presentation of genetic algorithm results
        #This is free (as you like) innovative part of the assessment.
        break;


if __name__ == '__main__':
    main()



