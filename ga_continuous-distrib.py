# -*- coding: utf-8 -*-
# Use standard python package only.
import random
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



from math import cos, pi, sin,sqrt

# MINIMUM GLOBAL VARIABLES TO BE USED
POPULATION_SIZE = 50  # Change POPULATION_SIZE to obtain better fitness.

GENERATIONS = 100 # Change GENERATIONS to obtain better fitness.
SOLUTION_FOUND = False

CROSSOVER_RATE = 0.8 # Change CROSSOVER_RATE  to obtain better fitness.
MUTATION_RATE = 0.6 # Change MUTATION_RATE to obtain better fitness.

# Variables for Testing
START_TESTS = True
CHANGED_VAR = "None"
DIFFERENCE = 0.2  # Refers to the difference (+) in CHANGED_VAR from previous run
START_VAL = 0  # Use if you wish to alter a CHANGED_VAR at the beginning of a run
NUM_OF_TESTS = 10

# Variables for functions and run types
LOWER_BOUND = -10
UPPER_BOUND = 10
FITNESS_CHOICE = 2
SELECTION_CHOICE = 1

CROSSOVER_CHOICE = 1
RUN_CHOICE = 1

# Can be altered to tune the genetic algorithm
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

# Generate an initial population
def generate_population(size, lower_bound, upper_bound):
    population = [generate_individual(NO_OF_GENES, lower_bound, upper_bound)
                  for _ in range(size)]
    # Returns an array of individuals, each composed of 1 or more ints
    return population

# Generate a single individual in the population
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


# Sum of Squares Function
def sum_square(individual):
    fitness = sum([(x+1) * individual[x]**2 for x in range(NO_OF_GENES)])
    # Case where fitness will be 0 and thus need to account for the error message
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

#Rosenbrock Function
def rosenbrock(individual):
    fitness = sum((100 * (individual[x+1] - individual[x]**2)**2 + ((individual[x] - 1) ** 2) for x in individual))
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


# Rastrigin Function
def rastrigin(individual):
    fitness = 10 * len(individual) + sum([((x - 1)**2 - 10 * cos(2 * pi * (x-1))) for x in individual])
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

# Dixon Price Function
def dixon_price(individual):
    fitness = (individual[0]-1)** 2 + sum((x + 1) * (2 * individual[x]**2 - individual[x-1])**2 for x in range(1, NO_OF_GENES))
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness


#Schwefel function
#TODO: Change the INPUT RANGE to be -500 - 500 for this one
def schwefel(individual):
    # Schwefel function defined as 418.9829 * dimensions - .... , here we assume 1 dimension
    fitness = 418 - sum(individual[x] * sin(sqrt(abs(individual[x]))) for x in range(1, NO_OF_GENES))
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

# Trid function
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


# Selects parents used to breed the next generation
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

    # Remove the weakest members of the pop and allow them to be replaced by children
    # Because function params for crossover takes num of off spring as POPULATION  - PARENTS
    if SELECTION_CHOICE == 2:
        index = np.argpartition(fitness, no_of_parents)
        for x in sorted(index[:no_of_parents], reverse=True):
            del population[x]
            del fitness[x]


    return parents

# Crossover genetic operator
def crossover(parents, num_of_offspring):

    offspring = np.empty((num_of_offspring, NO_OF_GENES))

    # Single Point Crossover - Just swaps one gene
    for i in range(num_of_offspring):
        parent1_index = i % NO_OF_PARENTS
        parent2_index = (i+1) % NO_OF_PARENTS

        if CROSSOVER_CHOICE == 1:
            if random.random() < CROSSOVER_RATE:
                offspring[i] = list(parents[parent1_index][0:4]) + list(parents[parent2_index][4:9])
            else:
                offspring[i] = list(parents[parent1_index])
        else:
            # 2 - Point Crossover - Performs a crossover at 2 poinst in the chromosome
            crossover_indexes = sorted(random.sample(NO_OF_GENES), 2)
            if random.random() < CROSSOVER_RATE:
                offspring[i] = list(parents[parent1_index][0:crossover_indexes[0]]) + \
                    list(parents[parent2_index][crossover_indexes[0]:crossover_indexes[1]]) + \
                    list(parents[parent1_index][crossover_indexes[1]::])
            else:
                offspring[i] = parents[parent1_index]
    return offspring


def mutation(offspring):
    # Random Resetting Mutation (Gives a random value from lower and upper bound parameters)
    no_of_mutations = 1
    if RUN_CHOICE == 1:
        for x in range(len(offspring-1)):
            if random.random() < MUTATION_RATE:
                random_index = random.sample(range(NO_OF_GENES), no_of_mutations)
                random_value = random.sample(range(LOWER_BOUND, UPPER_BOUND), no_of_mutations)
                for i in range(len(random_index)):
                    offspring[x][random_index[i]] = random_value[0]

    # Inversion Mutation, Flips - to + and + to -
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

# Returns the best individual form a population
def find_best_input(population, fitness):
    best_fitness = max(fitness)
    individual_index = fitness.index(best_fitness)

    return population[individual_index]

# Informational code for reading
def next_generation(generation, fitness, parents, offspring):
    best_fitness = max(fitness)
    print("\n Generation ", generation, ": ",
          "\n Parents Selected: \n", parents,
          "\n Offspring", offspring,
          "\nFitness", fitness,
          "\nBest fitness after generation", best_fitness)

# Function used to check if the optimal solution has been reached
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
    global saved_pop

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

    results_dict = {}
    gen_count = 1

    if RUN_NUM == 0:
        population = generate_population(POPULATION_SIZE, lower_bound, upper_bound)
        saved_pop = population.copy()
    elif RUN_NUM != 0 and (var_to_change == "POPULATION"):
        population = generate_population(POPULATION_SIZE, lower_bound, upper_bound)
    else:
        population = saved_pop.copy()

    fitness = [compute_fitness(x) for x in population]
    results_dict[gen_count] = max(fitness)

    if check_solution(population):
        print("Found after ", gen_count, "generations")
        SOLUTION_FOUND = True
    else:
        gen_count += 1

    while gen_count <= GENERATIONS and SOLUTION_FOUND != True:

        parents = selection(population, fitness, NO_OF_PARENTS)

        offspring = crossover(parents, POPULATION_SIZE - NO_OF_PARENTS)

        offspring = mutation(offspring)

        population = list(parents) + list(offspring)

        fitness = [compute_fitness(x) for x in population]

        if check_solution(population):
            SOLUTION_FOUND = True
            print("Found after ", gen_count, "generation(s)")
        else:
            gen_count += 1

    print("Best Individual", find_best_input(population, fitness))
    print("Outcome of best individual: ", max(fitness))

    if SOLUTION_FOUND:
        results_dict.pop(gen_count)
        SOLUTION_FOUND = False

    return results_dict


def main():
    global POPULATION_SIZE
    global GENERATIONS
    global SOLUTION_FOUND
    global CHANGED_VAR
    global NO_OF_GENES

    print("Global Run Settings: ")
    print("FITNESS_CHOICE: ", FITNESS_DICTIONARY[FITNESS_CHOICE])
    print("GENERATION_NUMBER: ", GENERATIONS)
    print("NUMBER OF GENES: ", NO_OF_GENES)
    print("INITIALISATION BOUNDARIES: ", LOWER_BOUND, " To ", UPPER_BOUND)

    if START_TESTS:
        results_dict = [dict() for _ in range(NUM_OF_TESTS)]
        for x in range(NUM_OF_TESTS):
            value = x * DIFFERENCE + START_VAL
            print("\nTEST RUN: ", x + 1, "\n")
            results_dict[x] = run(x, varToChange=CHANGED_VAR, value=value)
    else:
        results_dict = [dict()]
        results_dict[0] = run(0)

    # Borrowed code for graph
    # Prints all the runs, on a fitness against number of generations graph
    max_generations = range(1, GENERATIONS + 1)
    plt.figure(1)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    for x in range(len(results_dict)):
        values = results_dict[x].values()
        print(values)
        if START_TESTS:
            line_label = CHANGED_VAR + ": " + "{:.2f}".format(x * DIFFERENCE + START_VAL)
            plt.plot(max_generations[0:len(results_dict[x].keys())], values, label=line_label)
            # If the run got an ideal individual (as we remove that point)
            if len(results_dict[x].keys()) < GENERATIONS:
                plt.plot(max(results_dict[x].keys()), max(values), marker='o', markersize=5, color="black")
            if CHANGED_VAR != "NONE" and NUM_OF_TESTS <= 10: plt.legend()
        else:
            plt.plot(max_generations[0:len(results_dict[x].keys())], values)

    if START_TESTS:
        test_range = range(NUM_OF_TESTS)
        x_axis = [x * DIFFERENCE + START_VAL for x in test_range]
        if CHANGED_VAR == "NONE":
            x_axis = test_range
            CHANGED_VAR = "Run Number"
        # When points are infinite, its looks better to have the line at the
        # highest point in the graph and indicate infinite with a black dot.
        max_y = max([max(results_dict[x].values()) for x in test_range])

        # If the size of the results dict is less, we know that that run got
        # to infinite, therefore note their x coord
        inf_points = [x * DIFFERENCE + START_VAL for x in test_range if len(results_dict[x].keys()) < GENERATIONS]

        # Print the points for the results - if inf, plot it's y value as max_y
        fig2_y_axis = [max(results_dict[x].values()) if x * DIFFERENCE + START_VAL not in inf_points else max_y for x in
                       test_range]

        plt.figure(2)
        plt.xlabel(CHANGED_VAR)
        plt.ylabel("Fitness")
        plt.plot(x_axis, fig2_y_axis, color="green")
        plt.plot(x_axis, fig2_y_axis, marker='o', markersize=5, color="green")
        # Plot the inf markers
        for x in inf_points:
            plt.plot(x, max_y, marker='o', markersize=5, color="black")

    plt.show()

if __name__ == '__main__':
    main()



