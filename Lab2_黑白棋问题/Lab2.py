class Node:  # 定义节点类
    coefficient = 2

    def __init__(self, board, color, root_color, parent=None, pre_action=None):
        self.board = board
        self.color = color.upper()
        self.root_color = root_color
        self.parent = parent
        self.children = []
        self.best_child = None
        self.get_best_child()
        self.preAction = pre_action
        self.actions = list(self.board.get_legal_actions(color=color))
        self.isOver = self.game_over()
        self.reward = {'X': 0, 'O': 0}
        self.visit_count = 0
        self.value = {'X': 1e5, 'O': 1e5}
        self.isLeaf = True
        self.best_reward_child = None
        self.get_best_reward_child()

    def game_over(self):
        black_list = list(self.board.get_legal_actions('X'))
        white_list = list(self.board.get_legal_actions('O'))
        game_is_over = len(black_list) == 0 and len(white_list) == 0
        return game_is_over

    def get_value(self):
        if self.visit_count == 0:
            return
        for color in ['X', 'O']:
            self.value[color] = self.reward[color] / self.visit_count + \
                                Node.coefficient * math.sqrt(
                math.log(self.parent.visit_count) * 2 / self.visit_count)

    def add_child(self, child):
        self.children.append(child)
        self.get_best_child()
        self.get_best_reward_child()
        self.isLeaf = False

    def get_best_child(self):
        if len(self.children) == 0:
            self.best_child = None
        else:
            sorted_children = sorted(self.children, key=lambda child: child.value[self.color], reverse=True)
            self.best_child = sorted_children[0]
        return self.best_child

    def get_best_reward_child(self):
        if len(self.children) == 0:
            best_reward_child = None
        else:
            sorted_children = sorted(self.children, key=lambda child: child.reward[
                                                                          self.color] / child.visit_count if child.visit_count > 0 else -1e5,
                                     reverse=True)
            best_reward_child = sorted_children[0]
        self.best_reward_child = best_reward_child
        return self.best_reward_child


from copy import deepcopy
import csv
import torch
from func_timeout import func_timeout, FunctionTimedOut
import math
import os.path
import random


class MonteCarlo_Search:  # 主体部分，执行蒙特卡洛树搜索
    def __init__(self, board, color):
        self.root = Node(board=deepcopy(board), color=color, root_color=color)
        self.color = color
        self.experience = {"state": [], "reward": [], "color": []}
        self.max_experience = 10000000000
        self.trans = {"X": 1, "O": -1, ".": 0}
        self.learning_rate = 0.3
        self.epsilon = 0.3
        self.gamma = 0.999

    def get_experience(self):
        queue = []
        for child in self.root.children:
            queue.append(child)
        while len(queue) > 0:
            if len(self.experience) == self.max_experience:
                break
            if not queue[0].isLeaf:
                self.add_experiences(queue[0])
                for child in queue[0].children:
                    queue.append(child)
            queue.pop(0)

    def add_experiences(self, node: Node):

        if len(self.experience["reward"]) == self.max_experience:
            return

        experience = self.get_state(node)
        self.experience["state"].append(experience)
        reward = node.reward["X" if node.color == "O" else "O"] / node.visit_count
        self.experience["reward"].append(reward)
        self.experience["color"].append(node.color)

    def get_state(self, node):
        new_statement = node.board._board
        return new_statement

    def search(self):
        if len(self.root.actions) == 1:
            return self.root.actions[0]

        return self.search_by_montecarlo_tree()

    def search_by_montecarlo_tree(self):
        try:
            func_timeout(timeout=3, func=self.build_montecarlo_tree)
        except FunctionTimedOut:
            pass

        return self.root.get_best_reward_child().preAction

    def build_montecarlo_tree(self):
        while 1 == 1:
            current_node = self.select()
            if current_node.isOver:
                winner, difference = current_node.board.get_winner()
            else:
                if current_node.visit_count:
                    current_node = self.expand(current_node)
                winner, difference = self.simulation(current_node)
            self.back_propagation(node=current_node, winner=winner, difference=difference)

    def select(self):  # 选择
        current_node = self.root
        while not current_node.isLeaf:
            if random.random() > self.epsilon:
                current_node = current_node.get_best_child()
            else:
                current_node = random.choice(current_node.children)
            self.epsilon *= self.gamma
        return current_node

    def simulation(self, node: Node):  # 模拟
        board = deepcopy(node.board)
        color = node.color
        while not self.game_over(board=board):
            actions = list(board.get_legal_actions(color=color))
            if len(actions) != 0:
                board._move(random.choice(actions), color)
            color = 'X' if color == 'O' else 'O'
        winner, difference = board.get_winner()
        return winner, difference

    def expand(self, node: Node):  # 扩展
        if len(node.actions) == 0:
            board = deepcopy(node.board)
            color = 'X' if node.color == 'O' else 'O'
            child = Node(board=board, color=color, parent=node, pre_action="none", root_color=self.color)
            node.add_child(child)
            return node.best_child
        for action in node.actions:
            board = deepcopy(node.board)
            board._move(action=action, color=node.color)
            color = 'X' if node.color == 'O' else 'O'
            child = Node(board=board, color=color, parent=node, pre_action=action, root_color=self.color)
            node.add_child(child=child)
        return node.best_child

    def back_propagation(self, node: Node, winner, difference):  # 反向传播
        while node is not None:
            node.visit_count += 1
            if winner == 0:
                node.reward['O'] -= difference
                node.reward['X'] += difference
            elif winner == 1:
                node.reward['X'] -= difference
            elif winner == 2:
                pass
            if node is not self.root:
                node.parent.visit_count += 1
                for child in node.parent.children:
                    child.get_value()
                node.parent.visit_count -= 1
            node = node.parent

    def game_over(self, board):  # 判定游戏是否结束
        black_list = list(board.get_legal_actions('X'))
        white_list = list(board.get_legal_actions('O'))
        game_is_over = len(black_list) == 0 and len(white_list) == 0
        return game_is_over


class AIPlayer:
    def __init__(self, color: str):
        self.color = color.upper()
        self.comments = "请稍后，{}正在思考".format("黑棋(X)" if self.color == 'X' else "白棋(O)")

    def get_move(self, board):
        print(self.comments)
        action = MonteCarlo_Search(board, self.color).search()  # 执行蒙特卡洛树搜索，并返回对应的action
        return action