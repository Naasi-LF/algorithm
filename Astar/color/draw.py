import heapq
import matplotlib.pyplot as plt
import numpy as np
import random

# 定义地图，0表示通路，1表示墙壁
n = 20  # 地图的大小 n x n
grid = [[0 for _ in range(n)] for _ in range(n)]  # 创建 n x n 的二维列表，初始值为0

# 随机生成地图中的墙壁
for row in range(n):
    for col in range(n):
        if random.random() < 0.2:  # 20%的概率成为墙壁
            grid[row][col] = 1  # 设置为1表示墙壁

# 定义起点和终点坐标
start = (0, 0)  # 起点坐标，左上角 (行, 列)

# 随机生成终点坐标，且不能是墙壁，也不能与起点重合
while True:
    goal_row = random.randint(0, n - 1)
    goal_col = random.randint(0, n - 1)
    goal = (goal_row, goal_col)
    if grid[goal_row][goal_col] == 0 and goal != start:
        break  # 找到非墙壁的终点，退出循环

# 定义启发式函数，计算曼哈顿距离
def heuristic(a, b):
    row1, col1 = a  # 节点 a 的坐标 (行, 列)
    row2, col2 = b  # 节点 b 的坐标 (行, 列)
    return abs(row1 - row2) + abs(col1 - col2)  # 计算并返回曼哈顿距离

# 定义获取邻居节点的函数
def neighbors(node):
    row, col = node  # 当前节点的坐标 (行, 列)
    results = []  # 存储可通行的邻居节点
    # 定义四个方向的移动：上、下、左、右
    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_row, n_col = row + d_row, col + d_col  # 计算邻居节点的坐标
        # 检查新坐标是否在地图范围内且是否为通路
        if 0 <= n_row < n and 0 <= n_col < n and grid[n_row][n_col] == 0:
            results.append((n_row, n_col))  # 将可通行的邻居节点添加到结果列表
    return results  # 返回邻居节点列表

# 初始化绘图
fig, ax = plt.subplots(figsize=(8, 8))  # 创建绘图对象
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整图形边距

# 定义设置坐标轴的函数
def setup_axes():
    ax.set_xticks(np.arange(-0.5, n, 1))  # 设置 x 轴刻度
    ax.set_yticks(np.arange(-0.5, n, 1))  # 设置 y 轴刻度
    ax.set_xticklabels([])  # 移除 x 轴刻度标签
    ax.set_yticklabels([])  # 移除 y 轴刻度标签
    ax.grid(True)  # 显示网格线
    ax.set_xlim(-0.5, n - 0.5)  # 设置 x 轴范围
    ax.set_ylim(-0.5, n - 0.5)  # 设置 y 轴范围
    ax.set_aspect('equal')  # 设置坐标轴比例相等

setup_axes()  # 初始化坐标轴设置
ax.axis('off')  # 隐藏坐标轴边框
plt.ion()  # 打开交互模式，实时更新绘图

# 创建颜色映射矩阵，用于表示不同类型的格子
color_map = np.full((n, n), 'white', dtype=object)  # 初始化为白色
for row in range(n):
    for col in range(n):
        if grid[row][col] == 1:
            color_map[row][col] = 'red'  # 障碍物用红色表示

# 定义绘制网格的函数
def draw_grid():
    ax.clear()  # 清除当前绘图
    setup_axes()  # 重新设置坐标轴属性
    for row in range(n):
        for col in range(n):
            # 绘制每个格子，设置颜色和边框
            rect = plt.Rectangle([col - 0.5, n - row - 1.5], 1, 1,
                                 facecolor=color_map[row][col], edgecolor='black')
            ax.add_patch(rect)
            # 显示总代价 f(n) = g(n) + h(n)
            if (row, col) in cost_so_far:
                g = cost_so_far[(row, col)]  # 从起点到当前节点的实际代价 g(n)
                h_value = heuristic((row, col), goal)  # 启发式估计代价 h(n)
                f = g + h_value  # 总代价 f(n) = g(n) + h(n)
                ax.text(col, n - row - 1, f"{f}", ha='center', va='center', fontsize=6)  # 在格子中显示 f(n)
    # 标注起点和终点
    ax.text(start[1], n - start[0] - 1, 'Start', color='green', ha='center', va='center', fontweight='bold')
    ax.text(goal[1], n - goal[0] - 1, 'Goal', color='purple', ha='center', va='center', fontweight='bold')
    plt.pause(0.001)  # 暂停以更新绘图

# 实现带有平局决策的 A* 算法
def a_star_search(start, goal):
    frontier = []  # 创建优先队列，存储待探索的节点
    h_start = heuristic(start, goal)  # 计算起点的启发式值
    # 在优先队列中，元素为 (f(n), h(n), node)，这样在 f(n) 相同的情况下，会比较 h(n)
    heapq.heappush(frontier, (h_start, h_start, start))  # 将起点加入优先队列
    came_from = {}  # 记录每个节点的前驱节点
    global cost_so_far
    cost_so_far = {}  # 记录从起点到每个节点的实际代价 g(n)
    came_from[start] = None  # 起点没有前驱节点
    cost_so_far[start] = 0  # 起点的代价为 0

    while frontier:
        _, _, current = heapq.heappop(frontier)  # 从优先队列中取出代价最小的节点

        if current == goal:
            break  # 找到目标节点，结束搜索

        for neighbor in neighbors(current):
            new_cost = cost_so_far[current] + 1  # 计算邻居节点的实际代价
            # 如果邻居节点未被探索过，或发现了更优的路径
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost  # 更新邻居节点的代价
                h_value = heuristic(neighbor, goal)  # 计算邻居节点的启发式代价
                priority = new_cost + h_value  # 计算邻居节点的总优先级 f(n)
                # 在优先队列中加入 (f(n), h(n), neighbor)，以便在 f(n) 相同时比较 h(n)
                heapq.heappush(frontier, (priority, h_value, neighbor))
                came_from[neighbor] = current  # 记录邻居节点的前驱节点
                # 更新颜色为蓝色，表示已访问
                if neighbor != goal and neighbor != start:
                    color_map[neighbor[0]][neighbor[1]] = 'blue'
        # 绘制当前状态
        draw_grid()

    return came_from  # 返回节点的前驱关系

# 重建从起点到终点的路径
def reconstruct_path(came_from, start, goal):
    current = goal  # 从终点开始回溯
    path = []
    while current != start:
        path.append(current)  # 将当前节点添加到路径
        current = came_from.get(current)
        if current is None:
            # 如果无法到达终点（即 current 为 None），返回空路径
            return []
    path.append(start)  # 添加起点
    path.reverse()  # 反转路径，使其从起点到终点
    return path  # 返回完整路径

# 运行 A* 算法
came_from = a_star_search(start, goal)

# 绘制最终路径
path = reconstruct_path(came_from, start, goal)
if path:
    for node in path:
        if node != start and node != goal:
            color_map[node[0]][node[1]] = 'yellow'  # 路径用黄色表示
else:
    print("无法到达终点")

draw_grid()  # 绘制最终的网格
plt.ioff()  # 关闭交互模式
plt.show()  # 显示绘图
