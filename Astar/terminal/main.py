# main.py

from astar import a_star_search  # 从astar_search模块导入A*算法函数
from path import reconstruct_path  # 从reconstruct_path模块导入路径重建函数

# 定义起点和终点
start = (0, 0)  # 起点坐标 (行, 列)
goal = (4, 4)  # 终点坐标 (行, 列)

# 运行A*算法
came_from, cost_so_far = a_star_search(start, goal)  # 执行A*搜索，获取路径和成本

# 重建并打印最短路径
path = reconstruct_path(came_from, start, goal)  # 使用路径字典重建完整路径
print("找到的最短路径:")  # 输出提示信息
print(path)  # 打印最短路径
