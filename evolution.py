import random
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
from player import Player
import numpy as np
from config import CONFIG
from copy import deepcopy


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.max_fitness = 0
        self.max_fitness_list = []
        self.min_fitness_list = []
        self.avg_fitness_list = []
        self.generation_number = 0

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]
            if p.fitness > self.max_fitness:
                self.max_fitness = p.fitness

    def mutate(self, child: Player):

        # TODO
        # child: an object of class `Player`
        if self.mode == 'helicopter':
            child.nn.mutation_weights_with_a_probability(0.9, 0.5)
        elif self.mode == 'thrust':
            child.nn.mutation_weights_with_a_probability(0.9, 0.1)
        else:
            child.nn.mutation_weights_with_a_probability(0.9, 0.1)
        return child

    def cross_over(self, parent1: Player, parent2: Player):
        child1 = Player(self.mode)
        child2 = Player(self.mode)

        weights1, weights2 = parent1.nn.cross_over_weights(parent2.nn, 5)

        child1.nn.set_weights(*weights1)
        child2.nn.set_weights(*weights2)

        return child1, child2

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            new_players = []
            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # default
            i = 0
            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover
            cross_over_rate = 0.9
            while i < num_players:
                if random.random() < cross_over_rate:
                    sorted_list = sorted(random.choices(prev_players, k=2), key=lambda x: -x.fitness)
                    new_player_1, new_player_2 = self.cross_over(deepcopy(sorted_list[0]), deepcopy(sorted_list[1]))
                    new_players.append(self.mutate(new_player_1))
                    new_players.append(self.mutate(new_player_2))
                else:
                    sorted_list = sorted(random.choices(prev_players, k=5), key=lambda x: -x.fitness)
                    new_players.append(self.mutate(deepcopy(sorted_list[0])))
                    sorted_list = sorted(random.choices(prev_players, k=5), key=lambda x: -x.fitness)
                    new_players.append(self.mutate(deepcopy(sorted_list[0])))
                i += 2

            # new_players = deepcopy(prev_players)
            return new_players

    def next_population_selection(self, players: [Player], num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        selected_players = []
        i = 0
        while i < num_players:
            sorted_list = sorted(random.choices(players, k=5), key=lambda x: -x.fitness)
            selected_players.append(deepcopy(sorted_list[0]))
            i += 1
        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        avg_fitness = sum([p.fitness for p in players]) / len(players)
        max_fitness = max([p.fitness for p in players])
        min_fitness = min([p.fitness for p in players])

        self.max_fitness_list.append(max_fitness)
        self.min_fitness_list.append(min_fitness)
        self.avg_fitness_list.append(avg_fitness)
        self.generation_number += 1

        # draw figure

        return selected_players

    def plot(self):
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.title('red = max fitness\nblue=min fitness\nblack=avg fitness')
        plt.plot(range(self.generation_number), self.max_fitness_list, color='red')
        plt.plot(range(self.generation_number), self.min_fitness_list, color='blue')
        plt.plot(range(self.generation_number), self.avg_fitness_list, color='black')
