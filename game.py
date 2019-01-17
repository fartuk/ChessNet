import chess
from autogame import make_game
from chessnet import Engine

board = chess.Board()
eng = Engine('data/weights/best.h5')
engine = chess.uci.popen_engine('stockfish/stockfish-10-linux/Linux/stockfish_10_x64')
engine.setoption({'Threads':8, 'Skill':1})

result = make_game(eng, engine)
print(result[0])
#eng.position(board)
#str_move = eng.go()
#print(str_move)
