# reconstruct_path.py

# 重建从起点到终点的路径的函数
def reconstruct_path(came_from, start, goal):
    current = goal  # 从目标节点开始回溯
    path = []  # 存储路径的列表
    while current != start:  # 当未回溯到起点时
        path.append(current)  # 将当前节点添加到路径
        current = came_from[current]  # 移动到前驱节点
    path.append(start)  # 将起点添加到路径
    path.reverse()  # 反转路径，使其从起点到终点
    return path  # 返回完整的路径列表
