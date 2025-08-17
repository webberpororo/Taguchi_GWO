import numpy as np
import math
import pandas as pd

class GreyWolfOptimizer:
    def __init__(self, objective_function, num_dimensions, population_size, iterations):
        self.objective_function = objective_function
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.iterations = iterations

    def initialize_population(self):
        # return lower_bound + np.random.rand(self.population_size, self.num_dimensions) * (upper_bound - lower_bound)
        return np.random.rand(self.population_size, self.num_dimensions)

    def optimize_100(self, population):
        for i in range(self.iterations):
            a = 2 - 2 * i / self.iterations  # Eq. (3.6)
            for j in range(self.population_size):
                A1, A2, A3 = np.random.permutation(self.population_size)[:3]

                X1 = population[A1]
                X2 = population[A2]
                X3 = population[A3]
                D_alpha = np.abs(a * X1 - population[j])  # Eq. (3.2)
                D_beta = np.abs(a * X2 - population[j])  # Eq. (3.3)
                D_delta = np.abs(a * X3 - population[j])  # Eq. (3.4)
                X_new = (X1 + X2 + X3) / 3 - a * (D_alpha + D_beta + D_delta)  # Eq. (3.5)
                fitness_new = self.objective_function(X_new)
                fitness_current = self.objective_function(population[j])
                if fitness_new < fitness_current:
                    population[j] = X_new
        best_solutions = population[np.argmin([self.objective_function(ind) for ind in population])]
        return best_solutions

    def optimize(self, population, iter, total_iter):
        a = 2 - 2 * iter / total_iter  # Eq. (3.6)
        for j in range(self.population_size):
            A1, A2, A3 = np.random.permutation(self.population_size)[:3]
            X1 = population[A1]
            X2 = population[A2]
            X3 = population[A3]
            D_alpha = np.abs(a * X1 - population[j])  # Eq. (3.2)
            D_beta = np.abs(a * X2 - population[j])  # Eq. (3.3)
            D_delta = np.abs(a * X3 - population[j])  # Eq. (3.4)
            X_new = (X1 + X2 + X3) / 3 - a * (D_alpha + D_beta + D_delta)  # Eq. (3.5)
            fitness_new = self.objective_function(X_new)
            fitness_current = self.objective_function(population[j])
            if fitness_new < fitness_current:
                population[j] = X_new
        return population

# Example usage:
def sphere_function(x):
    return np.sum(np.array(x)**2)

def rastrigin(x):
    fitness_value = 0.0
    for i in range(len(x)):
        xi = x[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

def taguchi(data):
    col = [] #每exp計算平均值
    for i in range(32):
        total = 0
        for j in range(30):
            total = total + data.iat[i, j]
        col.append(total / 30)
    data['y'] = col
    print(data)

    l1 = [] #level1
    l2 = [] #level2
    for i in range(2, 32):
        level1 = 0 #the summary of level 1
        level2 = 0 #the summary of level 2

        for j in range(32):
            cols = data.iat[j, i - 2] #欄位值
            value = alpha[i] #當前1水準值
            if(cols == value):
                level1 = level1 + data.iat[j, 30]
            else:
                level2 = level2 + data.iat[j, 30]
        l1.append(level1 / 16)
        l2.append(level2 / 16)
    print("l1: ", l1)
    print("l2: ", l2)

    result = [] #最終優畫值
    for i in range(30):
        if(abs(l1[i]) < abs(l2[i])):
            result.append(l1[i])
        else:
            result.append(l2[i])
    print("result: ", result)
    return result

best_arr = []
dim = 30
iterations = 100
population_num = 50
fitness = rastrigin
optimizer = GreyWolfOptimizer(fitness, dim, population_num, iterations)
populations = optimizer.initialize_population()
best_solution = optimizer.optimize_100(populations)
best = fitness(best_solution)
arr = populations.copy()
for i in range(100):
    print("i: ", i)
    alpha = [0, 0, ]  # 存放每批次alpha最優
    beta = [0, 0, ]  # 存放每批次beta最優
    # if not np.array_equal(arr, populations):
    #     print("error")
    solution_1 = optimizer.optimize(populations, i, iterations)
    print("solution_1: ", solution_1)
    after_function = []
    for j in range(population_num):
        rst = fitness(solution_1[j])
        after_function.append(rst)
    sort = sorted(after_function)
    a = sort[0]
    b = sort[1]
    alpha_arr = solution_1[np.where(after_function == a)]
    beta_arr = solution_1[np.where(after_function == b)]
    for c in range(dim):
        alpha.append(alpha_arr[0][c])
        beta.append(beta_arr[0][c])
    data = pd.read_csv("D:\pycharmprojectnew\Taguchi_new.csv")
    for d in range(2, 32):
        data[f"{d}"] = data[f"{d}"].replace({1: alpha[d], 2: beta[d]})
    result = taguchi(data)
    print("alpha: ", alpha)
    print("beta: ", beta)
    print("1_best_solution", fitness(result))
    best_arr.append(fitness(result))
    print("best_solution", best_solution)
    if(fitness(best_solution) != best):
        print("error")
    fix_populations = []
    for e in range(population_num):
        rst2 = fitness(solution_1[e])
        fix_populations.append(rst2)
    sort1 = sorted(fix_populations)
    min_population = sort1[49]
    populations_arr = []
    for f in range(population_num):
        populations_arr.append(fitness(populations[f]))
    index = 0
    for g in range(population_num):
        if(populations_arr[g] == min_population):
            index = g

    populations[index] = result

for i in range(99):
    if(best_arr[i] < best_arr[i + 1]):
        print(i)
print("-------------------------------")
# for i in range(98):
#     if(part_100[i] < part_100[i + 1]):
#         print(i)
print("final_result: ", best_arr)
# print("part_100: ", part_100)
