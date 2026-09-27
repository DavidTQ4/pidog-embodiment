import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fluffy_games import TicTacToe


class TicTacToeTests(unittest.TestCase):
    def test_human_starts(self):
        self.assertEqual(TicTacToe().turn, "human")

    def test_occupied_square_is_rejected(self):
        game = TicTacToe()
        game.human_move(4)
        game.fluffy_move()
        with self.assertRaises(ValueError):
            game.human_move(4)

    def test_fluffy_cannot_be_forced_to_lose(self):
        def explore(game):
            if game.result:
                self.assertNotEqual(game.result, "X")
            elif game.turn == "fluffy":
                game.fluffy_move()
                explore(game)
            else:
                for cell, value in enumerate(game.board):
                    if not value:
                        branch = TicTacToe(list(game.board), game.turn, game.result)
                        branch.human_move(cell)
                        explore(branch)
        explore(TicTacToe())


if __name__ == "__main__":
    unittest.main()
