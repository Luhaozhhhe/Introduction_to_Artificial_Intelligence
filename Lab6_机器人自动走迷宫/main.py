# 导入相关包
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot # Keras版本
import matplotlib.pyplot as plt


def my_search(maze):

    move_map = {
        'u': (-1, 0), # 表示往上走
        'r': (0, +1), # 表示往右走
        'd': (+1, 0), # 表示往下走
        'l': (0, -1), # 表示往左走
    }

    class SearchTree(object):


        def __init__(self, loc=(), action='', parent=None):
            self.loc = loc  # 当前节点位置
            self.to_this_action = action  # 到达当前节点的动作
            self.parent = parent  # 当前节点的父节点
            self.children = []  # 当前节点的子节点

        def add_child(self, child):
            self.children.append(child)

        def is_leaf(self):
            return len(self.children) == 0
        
    def expand(maze, is_visit, node):
        child_number = 0  # 记录叶子节点个数
        can_move = maze.can_move_actions(node.loc)
        for a in can_move:
            new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
            if not is_visit[new_loc]:
                child = SearchTree(loc=new_loc, action=a, parent=node)
                node.add_child(child)
                child_number+=1
        return child_number  # 返回叶子节点个数
                
    def back_propagation(node):
        path = []
        while node.parent is not None:
            path.insert(0, node.to_this_action)
            node = node.parent
        return path

    def DFS(maze):
        start = maze.sense_robot()
        root = SearchTree(loc=start)
        queue = [root]  # 节点堆栈，用于层次遍历
        h, w, _ = maze.maze_data.shape
        is_visit = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
        path = []  # 记录路径
        temp = 0
        while True:
            current_node = queue[temp]  # 栈顶元素作为当前节点
            if current_node.loc == maze.destination:  # 到达目标点
                path = back_propagation(current_node)
                break

            if current_node.is_leaf() and is_visit[current_node.loc] == 0:  # 如果该点存在叶子节点且未拓展
                is_visit[current_node.loc] = 1  # 标记该点已拓展
                child_number = expand(maze, is_visit, current_node)
                temp+=child_number  # 开展一些列入栈操作
                for child in current_node.children:
                    queue.append(child)  # 叶子节点入栈
            else:
                queue.pop(temp)  # 如果无路可走则出栈
                temp-=1
                
        return path
    
    path = DFS(maze)
    return path

import random
from QRobot import QRobot

class Robot(QRobot):

     valid_action = ['u', 'r', 'd', 'l']

     def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon=0.5):
         self.maze = maze
         self.state = None
         self.action = None
         self.alpha = alpha
         self.gamma = gamma
         self.epsilon = epsilon  # 动作随机选择概率
         self.q_table = {}

         self.maze.reset_robot()  # 重置机器人状态
         self.state = self.maze.sense_robot()  # state为机器人当前状态

         if self.state not in self.q_table:  # 如果当前状态不存在，则为 Q 表添加新列
             self.q_table[self.state] = {a: 0.0 for a in self.valid_action}

     def train_update(self):

         self.state = self.maze.sense_robot()  # 获取机器人当初所处迷宫位置

         # 检索Q表，如果当前状态不存在则添加进入Q表
         if self.state not in self.q_table:
             self.q_table[self.state] = {a: 0.0 for a in self.valid_action}
         # action为机器人选择的动作
         action = random.choice(self.valid_action) if random.random() < self.epsilon else max(self.q_table[self.state], key=self.q_table[self.state].get) 

         reward = self.maze.move_robot(action)  # 以给定的方向移动机器人,reward为迷宫返回的奖励值
         next_state = self.maze.sense_robot()  # 获取机器人执行指令后所处的位置

         # 检索Q表，如果当前的next_state不存在则添加进入Q表
         if next_state not in self.q_table:
             self.q_table[next_state] = {a: 0.0 for a in self.valid_action}

         # 更新 Q 值表
         current_r = self.q_table[self.state][action]
         update_r = reward + self.gamma * float(max(self.q_table[next_state].values()))
         self.q_table[self.state][action] = self.alpha * self.q_table[self.state][action] +(1 - self.alpha) * (update_r - current_r)
         # 衰减随机选择动作的可能性
         self.epsilon *= 0.5  

         return action, reward

     def test_update(self):
         self.state = self.maze.sense_robot()  # 获取机器人现在所处迷宫位置

         # 检索Q表，如果当前状态不存在则添加进入Q表
         if self.state not in self.q_table:
             self.q_table[self.state] = {a: 0.0 for a in self.valid_action}
        
         action = max(self.q_table[self.state],key=self.q_table[self.state].get)  # 选择动作
         reward = self.maze.move_robot(action)  # 以给定的方向移动机器人

         return action, reward
