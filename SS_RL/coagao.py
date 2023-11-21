import random

# 初始化工件的加工时间和位置
processing_times = [4, 6, 3, 7, 5, 2, 9, 8, 6, 5]

# 初始化蜜蜂数量和迭代次数
n_employees = 10
n_lookers = 10
n_scouts = 5
n_iterations = 100

# 初始化最优解和最优适应度
best_solution = None
best_fitness = float('inf')

# 初始化雇佣蜂阶段
employees = [[random.randint(0, 9)] for _ in range(n_employees)]

# 迭代优化
for iteration in range(n_iterations):
    # 观察蜂阶段
    for i in range(n_lookers):
        # 选择雇佣蜂
        employee_index = random.randint(0, n_employees - 1)
        employee = employees[employee_index]

        # 生成新解
        k = random.randint(0, len(employee) - 1)
        new_solution = employee.copy()
        new_solution[k] = random.randint(0, 9)

        # 计算新解的适应度
        fitness = max(sum(processing_times[:j + 1]) for j in new_solution)

        # 更新最优解
        if fitness < best_fitness:
            best_solution = new_solution
            best_fitness = fitness

    # 侦察蜂阶段
    for i in range(n_scouts):
        # 生成新解
        new_solution = [random.randint(0, 9) for _ in range(len(employees[i]))]

        # 计算新解的适应度
        fitness = max(sum(processing_times[:j + 1]) for j in new_solution)

        # 更新最优解
        if fitness < best_fitness:
            best_solution = new_solution
            best_fitness = fitness

        # 替换蜜蜂
        employees[i] = new_solution

# 打印最优解和最优适应度
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
