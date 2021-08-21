from Game import Game
from Global import SCREEN_HEIGHT, SCREEN_WIDTH, FRAME_RATE
import pygame


def main():
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()
    pygame.init()

    size = (SCREEN_WIDTH, SCREEN_HEIGHT)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Tim's Terrific Title")
    clock = pygame.time.Clock()
    done = False

    game = Game()
    
    while not done:
        done = game.process_events()

        game.update()
        game.draw_screen(screen)

        pygame.display.flip()

        clock.tick(FRAME_RATE)
    pygame.quit()


if __name__ == "__main__":
    main()
    