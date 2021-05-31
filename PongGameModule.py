import math
import numpy as np


class PongGame:
    def __init__(self, width, height, velocity, initial_pos):
        self.width = width
        self.height = height
        self.velocity = velocity
        self.speed = np.array([velocity, velocity])
        self.pos = initial_pos
        self.collisionFlag = False
        self.collisionCounter = 0

    def move_ball(self):
        self.pos = np.add(self.pos, self.speed)

        if self.pos[0] > self.width or self.pos[0] < 0:
            self.speed[0] = self.speed[0] * (-1)
        if self.pos[1] > self.height or self.pos[1] < 0:
            self.speed[1] = self.speed[1] * (-1)

    def check_collision(self, pos_collider):
        if np.linalg.norm(self.pos-pos_collider) < 25 and not self.collisionFlag:
            self.collisionFlag = True
            self.speed = (normalize(np.add(self.pos, pos_collider*(-1)))*self.velocity).astype(int)

        if self.collisionFlag:              # check collision 10 frames after detects one
            self.collisionCounter += 1
            if self.collisionCounter > 20:
                self.collisionFlag = False
                self.collisionCounter = 0


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm
