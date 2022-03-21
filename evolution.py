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
        elif self.mode == 'gravity':
            child.nn.mutation_weights_with_a_probability(0.9, 0.5)
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
            # q_tournament
            new_players = self.q_tournament_selection_with_cross_over(num_players, prev_players)

            # SUS
            # new_players = self.sus_selection_with_cross_over(num_players, prev_players)

            return new_players

    def sus_selection_with_cross_over(self, num_players, prev_players):
        random.shuffle(prev_players)
        sum_fitness = sum([p.fitness for p in prev_players])
        probability_list = [p.fitness / sum_fitness for p in prev_players]

        x_prob = []
        for i, item in enumerate(probability_list):
            if i == 0:
                x_prob.append((0, item))
            else:
                x_prob.append((x_prob[i - 1][1], x_prob[i - 1][1] + item))

        first_selected_number = random.random() / num_players
        select_list = []
        select_list.append(first_selected_number)
        for i in range(1, num_players):
            select_list.append(select_list[i - 1] + 1 / num_players)
        new_players = []
        for n in select_list:
            selected = False
            for i, item in enumerate(x_prob):
                if selected is True:
                    break
                if item[0] <= n <= item[1]:
                    new_players.append(self.mutate(deepcopy(prev_players[i])))
                    selected = True

            assert selected is True
        assert len(new_players) == num_players

        return new_players

    def q_tournament_selection_with_cross_over(self, num_players, prev_players):
        new_players = []
        i = 0
        # a selection method other than `fitness proportionate`
        # implementing crossover
        # q-tournament
        cross_over_rate = 0.9
        if self.mode == 'gravity':
            cross_over_rate = 0.3
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
        return new_players

    def next_population_selection(self, players: [Player], num_players):

        # a selection method other than `top-k`
        # q_tournament
        selected_players = []
        i = 0
        k = 5
        if self.mode == 'gravity':
            k = 3
        while i < num_players:
            sorted_list = sorted(random.choices(players, k=k), key=lambda x: -x.fitness)
            selected_players.append(deepcopy(sorted_list[0]))
            i += 1
        #  plotting
        avg_fitness = sum([p.fitness for p in players]) / len(players)
        max_fitness = max([p.fitness for p in players])
        min_fitness = min([p.fitness for p in players])

        self.max_fitness_list.append(max_fitness)
        self.min_fitness_list.append(min_fitness)
        self.avg_fitness_list.append(avg_fitness)
        self.generation_number += 1

        return selected_players

    def plot(self):
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.title('red = max fitness\nblue=min fitness\nblack=avg fitness')
        plt.plot(range(self.generation_number), self.max_fitness_list, color='red')
        plt.plot(range(self.generation_number), self.min_fitness_list, color='blue')
        plt.plot(range(self.generation_number), self.avg_fitness_list, color='black')
