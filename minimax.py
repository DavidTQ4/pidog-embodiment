import math

def print_board(board):
    symbols = [' ', 'X', 'O']
    print("\nBoard:")
    for i in range(0, 9, 3):
        row = ' | '.join(symbols[board[j]] for j in range(i, i+3))
        print(" " + row)
        if i < 6:
            print("---|---|---")

def check_winner(board):
    win_positions = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for pos in win_positions:
        line = [board[i] for i in pos]
        if line == [1, 1, 1]:
            return 1  # Player
        elif line == [2, 2, 2]:
            return 2  # AI
    if 0 not in board:
        return 0  # Draw
    return None

def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result is not None:
        if result == 2:
            return 1  # AI win
        elif result == 1:
            return -1  # Player win
        else:
            return 0  # Draw

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == 0:
                board[i] = 2
                score = minimax(board, depth + 1, False)
                board[i] = 0
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                score = minimax(board, depth + 1, True)
                board[i] = 0
                best_score = min(score, best_score)
        return best_score

def best_move(board):
    best_score = -math.inf
    move = -1
    for i in range(9):
        if board[i] == 0:
            board[i] = 2
            score = minimax(board, 0, False)
            board[i] = 0
            if score > best_score:
                best_score = score
                move = i
    return move

def play_game():
    board = [0] * 9
    print("You are X (1). AI is O (2). Board positions are 0-8 left to right, top to bottom.")
    player_first = input("Go first? (y/n): ").strip().lower() == 'y'

    while True:
        print_board(board)
        if player_first:
            move = int(input("Enter your move (0-8): "))
            if board[move] != 0:
                print("Invalid move.")
                continue
            board[move] = 1
        else:
            ai_move = best_move(board)
            print(f"AI chooses position {ai_move}")
            board[ai_move] = 2

        winner = check_winner(board)
        if winner is not None:
            print_board(board)
            if winner == 1:
                print("You win!")
            elif winner == 2:
                print("AI wins!")
            else:
                print("It's a draw!")
            break

        player_first = not player_first

if __name__ == "__main__":
    play_game()
