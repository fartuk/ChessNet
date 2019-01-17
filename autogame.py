import chess.uci
import numpy as np
from tqdm import tqdm
import chess.pgn

def expect_elo(R_a, R_b):
    return 1 / (1 + 10 ** ((R_b - R_a) / 400))
    
def update_elo(R_a, R_b, res):
    R_a_new = R_a + 30 * (res[0] - expect_elo(R_a, R_b))
    R_b_new = R_b + 30 * (res[1] - expect_elo(R_b, R_a))
    return R_a_new, R_b_new

def make_game(engine0, engine1):
    '''
    engines should have methods .position(board) 
                            and .go(movetime) returning [move]
    '''
    engines = [engine0, engine1]
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    
    k = 0
    iter_cnt = 0

    while not node.board().is_game_over() and iter_cnt < 1500:
        engines[k].position(node.board())
        str_move = engines[k].go(movetime=1)[0].__str__()
        node = node.add_main_variation(chess.Move.from_uci(str_move))
        k = 1 - k
        iter_cnt += 1
        
    result = [0, 0]
    if node.board().is_checkmate():
        winner = 1 - k
        result[1-k] = 1
        
    if node.board().is_stalemate() or node.board().is_fivefold_repetition() \
        or node.board().is_seventyfive_moves() or node.board().is_insufficient_material() or iter_cnt == 1500:
        result = [0.5, 0.5]
    
    return result, game


