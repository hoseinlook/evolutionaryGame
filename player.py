import pygame
import numpy as np

from box_list import BoxList
from nn import NeuralNetwork
from config import CONFIG


class Player():

    def __init__(self, mode, control=False):

        self.control = control  # if True, playing mode is activated. else, AI mode.
        self.pos = [100, 275]  # position of the agent
        self.direction = -1  # if 1, goes upwards. else, goes downwards.
        self.v = 0  # vertical velocity
        self.g = 9.8  # gravity constant
        self.mode = mode  # game mode

        # neural network architecture (AI mode)
        layer_sizes = self.init_network(mode)

        self.nn = NeuralNetwork(layer_sizes)
        self.fitness = 0  # fitness of agent

    def move(self, box_lists: [BoxList], camera, events=None):
        # print("BOX LIST X", box_lists[0].x)
        # print("BOX LIST GAP", box_lists[0].gap_mid)
        # print("DIrectioN ", self.direction)

        if len(box_lists) != 0:
            if box_lists[0].x - camera + 60 < self.pos[0]:
                box_lists.pop(0)

        mode = self.mode
        # if len(box_lists) >= 2:
        #     box1: BoxList = box_lists[0]
        #     box2: BoxList = box_lists[1]
        #     agent_position = [camera + self.pos[0], self.pos[1]]
        #     x = camera + self.pos[0]
        #     print([(box1.x - x) / 1000, box1.gap_mid / 500, (box2.x - x) / 1000, box2.gap_mid / 500,
        #            agent_position[1] / 500, self.v/5 ])

        # manual control
        if self.control:
            self.get_keyboard_input(mode, events)

        # AI control

        else:
            agent_position = [camera + self.pos[0], self.pos[1]]
            self.direction = self.think(mode, box_lists, agent_position, self.v)

            # game physics
        if mode == 'gravity' or mode == 'helicopter':
            self.v -= self.g * self.direction * (1 / 60)
            self.pos[1] += self.v

        elif mode == 'thrust':
            self.v -= 6 * self.direction
            self.pos[1] += self.v * (1 / 40)

        # collision detection
        is_collided = self.collision_detection(mode, box_lists, camera)

        return is_collided

    # reset agent parameters
    def reset_values(self):
        self.pos = [100, 275]
        self.direction = -1
        self.v = 0

    def get_keyboard_input(self, mode, events=None):

        if events is None:
            events = pygame.event.get()

        if mode == 'helicopter':
            self.direction = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.direction = 1

        elif mode == 'thrust':
            self.direction = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction = 1
            elif keys[pygame.K_DOWN]:
                self.direction = -1

        for event in events:
            if event.type == pygame.KEYDOWN:

                if mode == 'gravity' and event.key == pygame.K_SPACE:
                    self.direction *= -1

    def init_network(self, mode):

        # you can change the parameters below

        layer_sizes = None
        if mode == 'gravity':
            layer_sizes = [6, 20, 1]
        elif mode == 'helicopter':
            layer_sizes = [6, 20, 1]
        elif mode == 'thrust':
            layer_sizes = [6, 20, 1]
        return layer_sizes

    def think(self, mode, box_lists, agent_position, velocity):

        # TODO
        # mode example: 'helicopter'
        # box_lists: an array of `BoxList` objects
        # agent_position example: [600, 250]
        # velocity example: 7

        if len(box_lists) == 0 or len(box_lists) == 1:
            return 0

        if mode == 'thrust':
            direction = self.think_about_thrust_mode(box_lists, agent_position, velocity)

        elif mode == 'helicopter':
            direction = self.think_about_helicopter_mode(box_lists, agent_position, velocity)

        elif mode == 'gravity':
            direction = self.think_about_gravity_mode(box_lists, agent_position, velocity)
        else:
            direction = -1
        return direction

    def think_about_helicopter_mode(self, box_lists, agent_position, velocity):
        if len(box_lists) == 0 or len(box_lists) == 1:
            return 0
        box1: BoxList = box_lists[0]
        box2: BoxList = box_lists[1]
        x = agent_position[0]
        input_array = [(box1.x - x) / 1000, box1.gap_mid / 500, (box2.x - x) / 1000, box2.gap_mid / 500,
                       agent_position[1] / 500,
                       velocity / 5]

        nn_input = np.array(
            [input_array])
        # print(nn_input)
        # print(nn_input)
        out = self.nn.forward(nn_input.reshape(6, 1))
        out = out[0][0]
        if out > 0.5:
            direction = 1
        else:
            direction = -1
        # print("DIRECTION ", direction)
        return direction

    def think_about_gravity_mode(self, box_lists, agent_position, velocity):
        if len(box_lists) == 0 or len(box_lists) == 1:
            return 0
        box1: BoxList = box_lists[0]
        box2: BoxList = box_lists[1]
        x = agent_position[0]
        input_array = [(box1.x - x) / 1000, box1.gap_mid / 500, (box2.x - x) / 1000, box2.gap_mid / 500,
                       agent_position[1] / 500,
                       velocity / 5]

        nn_input = np.array(
            [input_array])
        # print(nn_input)
        # print(nn_input)
        out = self.nn.forward(nn_input.reshape(6, 1))
        out = out[0][0]
        if out > 0.5:
            direction = 1
        else:
            direction = -1
        # print("DIRECTION ", direction)
        return direction

    def think_about_thrust_mode(self, box_lists, agent_position, velocity):
        if len(box_lists) == 0 or len(box_lists) == 1:
            return 0
        box1: BoxList = box_lists[0]
        box2: BoxList = box_lists[1]
        x = agent_position[0]
        input_array = [(box1.x - x) / 1000, box1.gap_mid / 500, (box2.x - x) / 1000, box2.gap_mid / 500,
                       agent_position[1] / 500,
                       velocity / 300]

        nn_input = np.array(
            [input_array])
        # print(nn_input)
        nn_input = nn_input / np.max(np.abs(nn_input))
        # print(nn_input)
        out = self.nn.forward(nn_input.reshape(6, 1))
        out = out[0][0]
        if out > 0.5:
            direction = 1
        else:
            direction = -1
        # print("DIRECTION ", direction)
        return direction
        pass

    def collision_detection(self, mode, box_lists, camera):
        if mode == 'helicopter':
            rect = pygame.Rect(self.pos[0], self.pos[1], 100, 50)
        elif mode == 'gravity':
            rect = pygame.Rect(self.pos[0], self.pos[1], 70, 70)
        elif mode == 'thrust':
            rect = pygame.Rect(self.pos[0], self.pos[1], 110, 70)
        else:
            rect = pygame.Rect(self.pos[0], self.pos[1], 50, 50)
        is_collided = False

        if self.pos[1] < -60 or self.pos[1] > CONFIG['HEIGHT']:
            is_collided = True

        if len(box_lists) != 0:
            box_list = box_lists[0]
            for box in box_list.boxes:
                box_rect = pygame.Rect(box[0] - camera, box[1], 60, 60)
                if box_rect.colliderect(rect):
                    is_collided = True

        return is_collided
