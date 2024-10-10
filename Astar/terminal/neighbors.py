# neighbors.py

from grid import grid, n  # 导入地图和地图大小

# 定义获取邻居节点的函数
def neighbors(node):
    row, col = node  # 当前节点的坐标 (行, 列)
    results = []  # 存储可通行的邻居节点
    # 定义上下左右四个方向的移动
    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # d_row和d_col分别为行和列方向的增量
        n_row, n_col = row + d_row, col + d_col  # 计算邻居节点的坐标
        if 0 <= n_row < n and 0 <= n_col < n:  # 检查新坐标是否在地图范围内
            if grid[n_row][n_col] == 0:  # 检查该位置是否为通路（grid[行][列]）
                results.append((n_row, n_col))  # 将可通行的邻居节点添加到结果列表
    return results  # 返回邻居节点列表
