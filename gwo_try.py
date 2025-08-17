import numpy as np
import math
import matplotlib.pyplot as plt

class GreyWolfOptimizer:
    def __init__(self, function, lb, ub, dimension, population_size, iterations):
        self.alpha_position = [1, 4] # [1, 4]
        self.alpha_score = float("inf") # float("inf")

        self.beta_position = np.zeros(dimension)
        self.beta_score = float("inf")

        self.delta_position = np.zeros(dimension)
        self.delta_score = float("inf")

        self.lb = lb
        self.ub = ub
        self.dimension = dimension
        self.population_size = population_size
        self.iterations = iterations
        self.function = function

    def optimize(self):
        cnt = 0
        wolves = []
        # for it in range(self.population_size):
        #     wolve = []
        #     x_ub = 75
        #     y_ub = 230
        #     z_ub = 115
        #     x = np.random.uniform(0, 1) * (x_ub - self.lb) + self.lb
        #     y = np.random.uniform(0, 1) * (y_ub - self.lb) + self.lb
        #     z = np.random.uniform(0, 1) * (z_ub - self.lb) + self.lb
        #     wolve.append(x)
        #     wolve.append(y)
        #     wolve.append(z)
        #     wolves.append(wolve)

        wolves = np.random.uniform(0, 1, (self.population_size, self.dimension)) * (self.ub - self.lb) + self.lb
        # wolves = np.array(wolves)
        # print(wolves.shape)

        list_a = []
        for t in range(self.iterations):

            list_a.append(self.alpha_score)
            for i in range(self.population_size):
                cnt += 1
                # print(cnt)
                fitness = self.function(wolves[i, :])
                # print(fitness)

                if fitness < self.alpha_score: # 大小於相反
                    self.alpha_score = fitness
                    self.alpha_position = np.copy(wolves[i, :])

                if fitness > self.alpha_score and fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_position = np.copy(wolves[i, :])

                if fitness > self.alpha_score and fitness > self.beta_score and fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_position = np.copy(wolves[i, :])

            a = 2 - t * (2 / self.iterations)

            for i in range(self.population_size):
                r1 = np.random.rand(self.dimension) # random.rand(self.dimension)
                r2 = np.random.rand(self.dimension)
                # print(a)
                # print(r1)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                # print(np.array(self.alpha_position).shape)
                # print(np.array(wolves[i, :]).shape)

                D_alpha = abs(C1 * self.alpha_position - wolves[i, :])
                X1 = self.alpha_position - A1 * D_alpha

                r1 = np.random.rand(self.dimension)
                r2 = np.random.rand(self.dimension)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * self.beta_position - wolves[i, :])
                X2 = self.beta_position - A2 * D_beta

                r1 = np.random.rand(self.dimension)
                r2 = np.random.rand(self.dimension)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * self.delta_position - wolves[i, :])
                X3 = self.delta_position - A3 * D_delta

                wolves[i, :] = (X1 + X2 + X3) / 3

            if((t + 1) % 10 == 0):
                # print(f'{t + 1}: ', self.alpha_position)
                print(f'{t + 1}: ', self.alpha_score)

        # searchagents_no = 1000
        #
        # plt.figure(figsize=(9, 15))
        # plt.subplots_adjust(wspace=0.5)
        # plt.subplot(1, 2, 1)
        # plt.plot(self.alpha_position[0], self.alpha_position[1], 'oc', linewidth=6, label="Alpha")
        # plt.plot(self.beta_position[0], self.beta_position[1], 'oy', linewidth=6, label="Beta")
        # plt.plot(self.delta_position[0], self.delta_position[1], 'ob', linewidth=6, label="Delta")
        # for i in range(searchagents_no):
        #     if i == searchagents_no - 1:
        #         plt.plot(wolves[i, 0], wolves[i, 1], 'og', linewidth=5, label="current_X")
        #         break
        # plt.scatter(wolves[:, 0], wolves[:, 1], color='r', marker='.', linewidths=3)
        # # plt.plot(0,0,'*k',linewidth=8,)
        # plt.title("GWO,a=2 to 0")
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(np.arange(self.iterations), list_a, 'b-')
        # plt.xlabel("iteration")
        # plt.ylabel("best score")
        # plt.title("convergence_curve")
        # plt.show()
        # print(self.alpha_position)
        # print(self.beta_position)
        # print(self.delta_position)

        return self.alpha_position, self.alpha_score
def rastrigin(x):
    fitness_value = 0.0
    for i in range(len(x)):
        xi = x[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

def ackley(x):
    d = len(x)
    sum1 = -20 * np.exp(-0.2 * np.sqrt(sum(x**2) / d))
    sum2 = -np.exp(sum(np.cos(2 * math.pi * xi) for xi in x) / d)
    return sum1 + sum2 + 20 + np.exp(1)

def sphere(x):
    return np.sum(x**2)

def avg(x):
    rnd = np.random.randint(0, 2)
    if rnd == 1:
        return np.sum(x) / 3
    elif rnd == 0:
        return 0

fitness = rastrigin
for i in range(1):
    gwo = GreyWolfOptimizer(fitness, -5.12, 5.12, 2, 20, 140)
    best, score = gwo.optimize()
    print(best)
    print(score)