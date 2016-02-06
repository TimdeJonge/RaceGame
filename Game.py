from Vector import Vector
from Unit import Player
from Global import *
import math
import pygame
import levels


class Game(object):
    def __init__(self):
        # SETTINGS:
        self.debug = True
        self.level = "L"

        # RELEVANT VALUES
        self.counter = 0
        self.player = Player(Vector([100, 40]), Vector([0, -1.1]), 5, PLAYER_COLOUR)

        if self.level == "Baby Park":
            self.obstacle_list = levels.baby_park
        elif self.level == "Circles":
            self.obstacle_list = levels.circles
        elif self.level == "L":
            self.obstacle_list = levels.l
        elif self.level == "Clear":
            self.obstacle_list = levels.clear
        self.turn = "neutral"
        self.acceleration = False
        self.brake = False
        self.camera = self.player.position - Vector([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])

    def process_events(self):
        """Handles all user inputs. Returns boolean "done" """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.turn = "left"
                elif event.key == pygame.K_RIGHT:
                    self.turn = "right"
                elif event.key == pygame.K_UP:
                    self.acceleration = True
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.brake = True
                elif event.key == pygame.K_r:
                    self.player.restart(Vector([100, 100]), Vector([1, 1]))

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.turn = "neutral"
                elif event.key == pygame.K_UP:
                    self.acceleration = False
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.brake = False
        return False

    def update(self):
        self.counter += 1

        if self.turn == "right":
            self.player.turn(math.pi/120)
        elif self.turn == "left":
            self.player.turn(-math.pi/120)
        if self.acceleration:
            if self.player.speed.values != [0, 0]:
                self.player.accelerate()
        if self.brake:
            if self.player.speed.values != [0, 0]:
                self.player.brake()

        self.player.move(self.obstacle_list)
        # The camera is a bit of magic in how it works. Don't mess with it too much and all will be fine.
        # Just subtract self.camera from everything that needs to be drawn on screen and it will work.
        self.camera = (self.camera.scalar(19) +
                       self.player.position -
                       Vector([SCREEN_WIDTH/2, SCREEN_HEIGHT/2]) +
                       self.player.speed.scalar(40)).scalar(.05)


    @staticmethod
    def draw_time(screen, counter):
        """Draws time in the lower right corner
        Keyword arguments:
        :param screen       the screen on which to draw the time
        :param counter      the amount of frames past since the start of counting
        :type counter       int
        """
        font = pygame.font.SysFont('Console', 20, False, False)
        total_seconds = counter // FRAME_RATE
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        output_string = "Time: {0:02}:{1:02}".format(minutes, seconds)
        text = font.render(output_string, True, WHITE)
        screen.blit(text, [SCREEN_WIDTH - 150, SCREEN_HEIGHT - 20])

    def draw_debug(self, screen):
        font = pygame.font.SysFont('Console', 10, False, False)
        pygame.draw.circle(screen, BLACK, [600, 400], 0)
        standard_vector = Vector((math.cos(self.player.direction), math.sin(self.player.direction)))
        pygame.draw.line(screen, WHITE, (self.player.position - self.camera).values,
                                        (self.player.position + standard_vector.scalar(50) - self.camera).values)
        debug_string1 = "x_speed = " + str(self.player.speed.values[0])
        debug_string2 = "y_speed = " + str(self.player.speed.values[1])
        debug_string3 = "Total speed = " + str(math.sqrt(self.player.speed.values[0]**2 + self.player.speed.values[1]**2))
        debug_string4 = "Location = " + str(self.player.position)
        text1 = font.render(debug_string1, True, WHITE)
        text2 = font.render(debug_string2, True, WHITE)
        text3 = font.render(debug_string3, True, WHITE)
        text4 = font.render(debug_string4, True, WHITE)
        screen.blit(text1, [0, 0])
        screen.blit(text2, [0, 15])
        screen.blit(text3, [0, 30])
        screen.blit(text4, [0, 45])

    def draw_screen(self, screen):
        """Calls all draw methods."""
        screen.fill(BACKGROUND_COLOUR)
        for obstacle in self.obstacle_list:
            obstacle.draw(screen, self.camera)
        self.player.draw_player(screen, self.camera)
        Game.draw_time(screen, self.counter)
        if self.debug:
            self.draw_debug(screen)
