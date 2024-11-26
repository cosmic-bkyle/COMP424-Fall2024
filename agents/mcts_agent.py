from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("mcts_agent")
class MCTSAgent(Agent):
    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"
        
    def step(self, chess_board, player, opponent):
        start_time = time.time()
        time_limit = 1.8  # Allow some buffer for time management

        # Root of the tree is the current state
        root = Node(chess_board, player, opponent)

        while time.time() - start_time < time_limit:
            # Selection: Traverse the tree to find the most promising node
            node = self.select(root)

            # Expansion: Expand the node by creating child nodes for valid moves
            if not node.is_fully_expanded():
                node = self.expand(node)

            # Simulation: Simulate a game from the expanded node
            reward = self.simulate(node)

            # Backpropagation: Update the node and its ancestors with the result
            self.backpropagate(node, reward)

        # Choose the best move based on visit count
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def select(self, node):
        """Select a child node using the UCB1 formula."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = max(node.children, key=lambda child: child.get_uct())
        return node

    def expand(self, node):
        """Expand a node by creating a new child for one of its unexplored moves."""
        valid_moves = node.get_unexplored_moves()
        move = valid_moves.pop()
        child_board = deepcopy(node.board)
        execute_move(child_board, move, node.player)
        child_node = Node(
            board=child_board,
            player=node.opponent,
            opponent=node.player,
            parent=node,
            move=move
        )
        node.add_child(child_node)
        return child_node

    def simulate(self, node):
        """Simulate a random game from the given node."""
        current_board = deepcopy(node.board)
        current_player = node.player
        opponent = node.opponent

        while True:
            valid_moves = get_valid_moves(current_board, current_player)
            if valid_moves:
                move = valid_moves[np.random.randint(len(valid_moves))]
                execute_move(current_board, move, current_player)
            else:
                current_player, opponent = opponent, current_player
                valid_moves = get_valid_moves(current_board, current_player)
                if not valid_moves:
                    break

            current_player, opponent = opponent, current_player

        _, player_score, opponent_score = check_endgame(current_board, node.player, node.opponent)
        return 1 if player_score > opponent_score else 0 if player_score < opponent_score else 0.5

    def backpropagate(self, node, reward):
        """Backpropagate the simulation result through the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

class Node:
    def __init__(self, board, player, opponent, parent=None, move=None):
        self.board = board
        self.player = player
        self.opponent = opponent
        self.parent = parent
        self.move = move
        self.children = []
        self.unexplored_moves = get_valid_moves(board, player)
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.unexplored_moves) == 0

    def is_terminal(self):
        return not self.unexplored_moves and not get_valid_moves(self.board, self.opponent)

    def add_child(self, child):
        self.children.append(child)

    def get_unexplored_moves(self):
        return self.unexplored_moves

    def get_uct(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)
