import random

from player import Player
import numpy as np
from config import CONFIG
from copy import deepcopy


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child: Player):

        # TODO
        # child: an object of class `Player`
        child.nn.mutation_weights_with_a_probability(0.9)
        return child

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
            while i < num_players:
                sorted_list = sorted(random.choices(prev_players, k=10), key=lambda x: -x.fitness)
                new_players.append(self.mutate(deepcopy(sorted_list[0])))
                i += 1

                # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

            # new_players = deepcopy(prev_players)
            return new_players

    def next_population_selection(self, players: [Player], num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        selected_players=[]
        i=0
        while i < num_players:
            sorted_list = sorted(random.choices(players, k=3), key=lambda x: -x.fitness)
            selected_players.append(deepcopy(sorted_list[0]))
            i += 1
        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        return selected_players
