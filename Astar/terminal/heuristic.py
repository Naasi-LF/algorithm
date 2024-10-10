# heuristic.py

# 定义计算曼哈顿距离的函数
def heuristic(a, b):
    row1, col1 = a  # 节点a的坐标 (行, 列)
    row2, col2 = b  # 节点b的坐标 (行, 列)
    return abs(row1 - row2) + abs(col1 - col2)  # 计算曼哈顿距离并返回
