# astar_search.py

import heapq  # 导入堆队列模块，提供优先队列的功能
from neighbors import neighbors  # 从neighbors模块导入neighbors函数
from heuristic import heuristic  # 从heuristic模块导入heuristic函数

# 实现A*算法的函数
def a_star_search(start, goal):
    frontier = []  # 创建一个优先队列，存储待探索的节点
    heapq.heappush(frontier, (0, start))  # 将起点加入队列，优先级为0
    came_from = {}  # 记录路径的字典，键为节点，值为前驱节点
    cost_so_far = {}  # 记录从起点到当前节点的成本
    came_from[start] = None  # 起点的前驱节点为None
    cost_so_far[start] = 0  # 起点的成本为0
    while frontier:  # 当队列不为空时，继续搜索
        _, current = heapq.heappop(frontier)  # 取出优先级最低的节点，只需要节点坐标
        if current == goal:  # 如果到达目标节点，结束搜索
            break  # 退出循环
        for neighbor in neighbors(current):  # 遍历当前节点的所有邻居
            new_cost = cost_so_far[current] + 1  # 计算从起点到邻居节点的成本（假设移动成本为1）
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:  # 如果邻居未被探索过或找到更低成本的路径
                cost_so_far[neighbor] = new_cost  # 更新邻居节点的成本
                priority = new_cost + heuristic(goal, neighbor)  # 计算邻居节点的优先级（成本 + 预估距离）
                heapq.heappush(frontier, (priority, neighbor))  # 将邻居节点加入队列
                came_from[neighbor] = current  # 记录邻居节点的前驱节点
    return came_from, cost_so_far  # 返回路径字典和成本字典
