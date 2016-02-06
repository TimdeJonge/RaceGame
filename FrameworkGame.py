import pygame


def main():
    pygame.init()

    size = (700, 500)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Tim's Testing Title")

    clock = pygame.time.Clock()
    frame_rate = 100
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("User asked to quit.")
                done = True
        # ALL CODE SHOULD GO ABOVE THIS COMMENT
        pygame.draw.circle(screen, (255, 255, 255), [20, 20], 5)
        pygame.display.flip()

        clock.tick(frame_rate)
    pygame.quit()

if __name__ == "__main__":
    main()
