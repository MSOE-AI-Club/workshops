"""The game window: draws the board and reads the keys. The rules are in game.py."""

import argparse

import pygame

from snake.game import DOWN, LEFT, RIGHT, UP, Game

CELL = 28
BACKGROUND = (12, 10, 24)
GRID = (28, 24, 48)
SNAKE_HEAD = (170, 110, 255)
SNAKE_BODY = (110, 70, 210)
FOOD = (80, 200, 255)
TEXT = (235, 235, 245)

KEYS = {
    pygame.K_UP: UP,
    pygame.K_w: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_d: RIGHT,
}


def draw(screen: pygame.Surface, font: pygame.font.Font, game: Game) -> None:
    screen.fill(BACKGROUND)
    for x in range(game.width):
        for y in range(game.height):
            pygame.draw.rect(screen, GRID, (x * CELL, y * CELL, CELL, CELL), 1)
    # TODO: draw the food, the snake, the score, and a game over message.
    pygame.display.flip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="snake", description="Snake, for the MSOE AI Club.")
    parser.add_argument("--speed", type=int, default=8, help="moves per second (default: 8)")
    args = parser.parse_args(argv)

    pygame.init()
    game = Game()
    screen = pygame.display.set_mode((game.width * CELL, game.height * CELL))
    pygame.display.set_caption("Snake")
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return 0
            # TODO: arrow keys and WASD steer (see KEYS). R starts a new game after a game over.
        # TODO: move the snake one step each tick.
        draw(screen, font, game)
        clock.tick(args.speed)
