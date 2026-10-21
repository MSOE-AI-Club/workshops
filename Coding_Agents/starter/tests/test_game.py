from snake.game import Game


def test_board_is_20_by_15():
    game = Game()
    assert (game.width, game.height) == (20, 15)
