import numpy as np
import pandas as pd
import os
import chess
import chess.pgn
from board_utils import get_position_tensor, flip_move, get_position, move2tensor, ohe_board


def keras_generator(batch_size=32, min_elo=2900):
    while(1):
        pgns_path = 'data/pgns/40_40/'
        pgn_name = np.random.choice(os.listdir(pgns_path))  
        
        pgn = open(pgns_path + pgn_name)
        game = chess.pgn.read_game(pgn)
        while game is not None:
            x_batch = []
            y_batch = []
        
            if int(game.headers['BlackElo']) > min_elo and int(game.headers['WhiteElo']) > min_elo:
                white = np.random.choice([False, True])
            elif max(int(game.headers['BlackElo']), int(game.headers['WhiteElo'])) > min_elo:
                white = bool(np.argmax([int(game.headers['BlackElo']), int(game.headers['WhiteElo'])]))
            else:
                break
                
            for move_id in np.random.choice(list(range(1 - int(white), int(game.headers['PlyCount']) - 1, 2)), batch_size):
                position = get_position(game, move_id)
                move_str = position.move.__str__()
                feat_tensor = get_position_tensor(position.parent.board(), white=white)
                if not white:
                    move_str = flip_move(move_str)

                move_tensor = move2tensor(move_str)
                assert move_tensor.sum() == 1
                
                x_batch += [feat_tensor]
                y_batch += [move_tensor]
                
            game = chess.pgn.read_game(pgn)


            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            yield x_batch, y_batch

        
