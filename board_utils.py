import numpy as np
import pandas as pd
import chess
import chess.pgn

def str2coord(move):
    trantab = str.maketrans('abcdefgh', '01234567')
    return 8 - int(move[1]), int(move[0].translate(trantab))

def coord2str(y, x):
    trantab = str.maketrans('01234567', 'abcdefgh')
    return str(x).translate(trantab) + str(8 - y)

def ohe_board(board, white=True):
    if white:
        trantab = str.maketrans('BKNPQRbknpqr.', '0123456666666')
    else:
        trantab = str.maketrans('bknpqrBKNPQR.', '0123456666666')
        
    ohe_mapa = np.array([[1., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 1.],
                         [0., 0., 0., 0., 0., 0.]])
    text_board = board.__str__()
    int_flatten_arr = np.array(text_board.translate(trantab).replace('\n', ' ').split(' ')).astype('int')
    return ohe_mapa[int_flatten_arr].reshape((8,8,6))

def flip_move(move_str):
    '''
    Example: e2e4 -> e7e5
    '''
    y0, x0 = str2coord(move_str[:2])
    y1, x1 = str2coord(move_str[2:4])

    y0 = 7 - y0
    y1 = 7 - y1

    move_str = coord2str(y0, x0) + coord2str(y1, x1)
    return move_str

def get_position(game, n):
    '''
    Input: game - chess Game
           n -move number
    Return: chess position
    '''
    k = 0
    for i in game.mainline():
        if k == n:
            break
        k += 1
    return i

def get_position_tensor(board, white=True):
    '''
    Input: game - chess Game
           white - for white player
    Return: 3D-np.array representation of board position
    '''
    white_ohe = ohe_board(board, white=True)
    black_ohe = ohe_board(board, white=False)
    if white:
        feat_tensor = np.concatenate([white_ohe, black_ohe], axis=-1)
    else:
        feat_tensor = np.concatenate([black_ohe, white_ohe], axis=-1)
        feat_tensor = np.flip(feat_tensor, axis=0)
    return feat_tensor


def move2tensor(move_str):
    move_from_str, move_to_str = move_str[:2], move_str[2:4]
    move_from = str2coord(move_from_str)
    move_to = str2coord(move_to_str)

    move_tensor = np.zeros((8,8,81))
    result = np.zeros(81)
    knife_mapa = {(2, 1): 0, (2, -1): 1, (1, 2): 2, (1, -2): 3, (-1, 2): 4, (-1, -2): 5, (-2, 1): 6, (-2, -1): 7}
    diff_x = move_to[1] - move_from[1]
    diff_y = move_to[0] - move_from[0]
    # Vertical move
    if diff_x == 0:
        move_type = 0
        result[move_type * 8 + move_to[0]] = 1
    # Horizontal move
    elif diff_y == 0:
        move_type = 1
        result[move_type * 8 + move_to[1]] = 1
    # First diagonal move
    elif diff_x == diff_y:
        move_type = 2
        result[move_type * 8 + move_to[1]] = 1
    # Second diagonal move
    elif diff_x == - diff_y:
        move_type = 3
        result[move_type * 8 + move_to[1]] = 1
    # Knife move
    elif np.abs(diff_x * diff_y) == 2:
        move_type = 4
        result[move_type * 8 + knife_mapa[diff_x, diff_y]] = 1

    move_tensor[move_from[0], move_from[1], :] = result
    return move_tensor

def tensor2move(move_tensor, white=True):
    y, x, n = np.unravel_index(move_tensor.argmax(), move_tensor.shape)
    knife_mapa = {0:(2, 1), 1:(2, -1), 2:(1, 2), 3:(1, -2), 4:(-1, 2), 5:(-1, -2), 6:(-2, 1), 7:(-2, -1)}
    move_from_str = coord2str(y, x)
    move_type = n // 8
    # Vertical move
    if move_type == 0:
        move_to_str = coord2str(n % 8, x)
    # Horizontal move
    elif move_type == 1:
        move_to_str = coord2str(y, n % 8)
    # First diagonal move
    elif move_type == 2:
        move_to_str = coord2str(y + (n % 8 - x), n % 8)
    # Second diagonal move
    elif move_type == 3:
        move_to_str = coord2str(y - (n % 8 - x), n % 8)
    # Knife move
    elif move_type == 4:
        diff_x, diff_y = knife_mapa[n % 8]
        move_to_str = coord2str(y + diff_y, x + diff_x)
    
    move_str = move_from_str + move_to_str
    if not white:
        move_str = flip_move(move_str)
    
    return move_str
