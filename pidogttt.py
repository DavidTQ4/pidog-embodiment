import subprocess
import time
from preset_actions import *
from function_demonstration import *

# Simple voice wrapper
def speak(text):
    print(text)
    subprocess.run(['espeak', text])

# Tic Tac Toe board
board = [' ' for _ in range(9)]

def print_board():
    print()
    for i in range(3):
        print(f"{board[i*3]} | {board[i*3+1]} | {board[i*3+2]}")
        if i < 2:
            print("--+---+--")
    print()

def is_winner(brd, player):
    win_combos = [(0,1,2), (3,4,5), (6,7,8),
                  (0,3,6), (1,4,7), (2,5,8),
                  (0,4,8), (2,4,6)]
    return any(all(brd[i] == player for i in combo) for combo in win_combos)

def is_draw(brd):
    return ' ' not in brd

def minimax(brd, depth, is_maximizing):
    if is_winner(brd, 'O'): return 1
    if is_winner(brd, 'X'): return -1
    if is_draw(brd): return 0

    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if brd[i] == ' ':
                brd[i] = 'O'
                score = minimax(brd, depth + 1, False)
                brd[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if brd[i] == ' ':
                brd[i] = 'X'
                score = minimax(brd, depth + 1, True)
                brd[i] = ' '
                best_score = min(score, best_score)
        return best_score

def best_move():
    best_score = -float('inf')
    move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, 0, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    return move

# PiDog gets ready
my_dog.do_action('sit', speed=80)
time.sleep(1)
speak("Let's play Tic Tac Toe. I'm ready!")

# Game loop
while True:
    print_board()

    # Human move
    move = int(input("Enter your move (1-9): ")) - 1
    if board[move] != ' ':
        print("Invalid move.")
        continue
    board[move] = 'X'

    if is_winner(board, 'X'):
        print_board()
        speak("You win this time. Woof!")
        break
    if is_draw(board):
        print_board()
        speak("That was a tough game. It's a draw.")
        break

    # AI move
    move = best_move()
    board[move] = 'O'
    print(f"PiDog plays move {move + 1}")

    if is_winner(board, 'O'):
        print_board()
        my_dog.speak('howl', volume=10)
        speak("I win! Better luck next time, human!")
        my_dog.do_action('stretch', speed=80)
        break
    if is_draw(board):
        print_board()
        speak("That was a tough game. It's a draw.")
        break
