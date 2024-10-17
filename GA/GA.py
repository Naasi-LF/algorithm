import numpy as np
import matplotlib.pyplot as plt
 
# 定义目标函数，将学号融入其中
# 学号 22030531 -> x^2 + 2.2030531 
def fitness(x):
    return x ** 2 +  2.2030531   # 22030531 融入目标函数
 
# 初始化种群
# 生成一个大小为 size 的种群，每个个体的值在 bounds 范围内。生成的是一个二维数组，每个个体是一个一维数组
def initialize_population(size, bounds):
    return np.random.uniform(bounds[0], bounds[1], (size, 1))
 
# 选择个体（轮盘赌选择）
def selection(pop, fitness_values):
    total_fitness = np.sum(fitness_values)  # 所有个体的总适应度值
    probabilities = fitness_values / total_fitness  # 每个个体被选择的概率
    indices = np.random.choice(len(pop), size=len(pop), p=probabilities.flatten())  # 根据这些概率随机选择个体，生成新的种群
    return pop[indices]
 
# 交叉操作（单点交叉）
def crossover(pop, crossover_rate=0.7):
    offspring = []
    for i in range(0, len(pop), 2):  # 以步长为2遍历种群
        parent1, parent2 = pop[i], pop[i + 1]  # 随机选择两个父代
        if np.random.rand() < crossover_rate and parent1.size > 1:  # 如果随机数小于交叉概率且父代长度大于1
            cross_point = np.random.randint(1, parent1.size)  # 选择一个交叉点
            child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])  # 在该点进行单点交叉生成两个子代
            child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
        else:  # 如果不满足交叉条件，直接复制父代作为子代。
            child1, child2 = parent1, parent2
        offspring.append(child1)
        offspring.append(child2)
    return np.array(offspring)
 
# 变异操作（简单变异）
def mutation(pop, mutation_rate=0.01, bounds=(-10, 10)):
    for i in range(len(pop)):
        if np.random.rand() < mutation_rate:  # 如果随机数小于变异概率
            mut_point = np.random.randint(0, pop[i].size)  # 随机选择该个体中的一个基因位置
            pop[i][mut_point] = np.random.uniform(bounds[0], bounds[1])
    return pop
 
# 主遗传算法流程
def genetic_algorithm(pop_size, bounds, num_generations, crossover_rate, mutation_rate):
    # 初始化种群
    pop = initialize_population(pop_size, bounds)
    best_solutions = []
 
    for gen in range(num_generations):
        # 计算适应度
        fitness_values = fitness(pop)
 
        # 选择
        selected_pop = selection(pop, fitness_values)
 
        # 交叉
        offspring = crossover(selected_pop, crossover_rate)
 
        # 变异
        pop = mutation(offspring, mutation_rate, bounds)
 
        # 保存当前代的最佳个体
        best_solutions.append(np.min(fitness_values))
 
    return pop, best_solutions
 
# 参数设置
# 学号：22030531
pop_size = 3100  # 种群大小, 将学号前四位作为种群大小：2203
bounds = (-5, 5)  # 每个个体的取值范围，将05用于设置区间：(-5,5)
num_generations = 31  # 将学号中 "31" 作为遗传代数：53
# 所谓遗传代数，就是迭代次数

crossover_rate = 0.8  # 交叉概率
mutation_rate = 0.1  # 变异概率

# 参数设置
# pop_size = 5000  # 种群大小  增大种群规模: 5000->100000
# bounds = (-10, 10)  # 每个个体的取值范围
# num_generations = 100  # 遗传算法运行的代数 增加迭代次数: 100->200
# crossover_rate = 0.9  # 交叉概率
# mutation_rate = 0.05  # 变异概率  增加变异率以增加多样性: 0.01->0.05
 
# 运行遗传算法
final_pop, best_solutions = genetic_algorithm(pop_size, bounds, num_generations, crossover_rate, mutation_rate)
 
# 绘制结果
plt.plot(best_solutions)
plt.xlabel('Generation')
plt.ylabel('Fitness (Minimized Value of x^2 + 2.2030531)')
plt.title('Genetic Algorithm Optimization with Student ID Influence')
plt.show()
 
# 输出最终的最佳解
best_individual = final_pop[np.argmin(fitness(final_pop))]
print("Best solution found:", best_individual)
print("Fitness of the best solution:", fitness(best_individual))
