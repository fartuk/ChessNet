from model import get_model
from board_utils import tensor2move, get_position_tensor
import chess
import numpy as np

class Engine:
    def __init__(self, path):
        self.model = get_model()
        self.model.load_weights(path)
        self.board = None
        
    def position(self, board):
        self.board = board
       
    def go(self, movetime):
        feat_tensor = get_position_tensor(self.board, white=self.board.turn)
        ohe = np.expand_dims(feat_tensor, axis=0)
        move_tensor = self.model.predict(ohe)[0]
        move = chess.Move.from_uci(tensor2move(move_tensor, white=self.board.turn))
        while move not in self.board.legal_moves:
            y, x, n = np.unravel_index(move_tensor.argmax(), move_tensor.shape)
            move_tensor[y, x, n] = 0
            move = chess.Move.from_uci(tensor2move(move_tensor, white=self.board.turn))

        return [move]