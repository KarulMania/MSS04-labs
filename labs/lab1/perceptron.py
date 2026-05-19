import numpy as np
import matplotlib.pyplot as plt


def H(z):
    if z >= 0:
        return 1
    else:
        return 0


class Perceptron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = np.zeros(n_inputs)
        self.bias = 0

    def out(self, X):
        z = np.dot(self.weights, X) + self.bias
        return H(z)

    def learn(self, data_learn, eta, epochs):
        error_list = []  # to track errors per epoch
        for epoch in range(epochs):
            for X, y in data_learn:
                y_pred = self.out(X)
                error = y - y_pred
                if error != 0:
                    error += 1  # update error counter
                if y == 1 and y_pred == 0:
                    self.weights = self.weights + eta * error * X
                    self.bias = self.bias + eta * error
                elif y == 0 and y_pred == 1:
                    self.weights = self.weights - eta * error * X
                    self.bias = self.bias - eta * error
            print(f"End of epoch {epoch + 1}, weights: {self.weights}, bias: {self.bias}")
        error_list.append(error)
        print(f"Epoch {epoch + 1}/{epochs}, Error: {error}")
        return error_list


def plot_decision_boundary(perceptron, X, Y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([perceptron.out(point) for point in grid])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])


Y_OR = np.array([1, 1, 1, 0])
Y_AND = np.array([1, 0, 0, 0])
Y_NAND = np.array([0, 1, 1, 1])


data_learn_OR = list(zip(X, Y_OR))
data_learn_AND = list(zip(X, Y_AND))
data_learn_NAND = list(zip(X, Y_NAND))


p_or = Perceptron(n_inputs=2)
p_or.learn(data_learn_OR, eta=0.1, epochs=10)

print("OR predictions:")
for x in X:
    print(x, "->", p_or.out(x))

plot_decision_boundary(p_or, X, Y_OR, "OR Decision Boundary")


p_and = Perceptron(n_inputs=2)
p_and.learn(data_learn_AND, eta=0.1, epochs=10)

print("AND predictions:")
for x in X:
    print(x, "->", p_and.out(x))

plot_decision_boundary(p_and, X, Y_AND, "AND Decision Boundary")


p_nand = Perceptron(n_inputs=2)
p_nand.learn(data_learn_NAND, eta=0.1, epochs=10)

print("NAND predictions:")
for x in X:
    print(x, "->", p_nand.out(x))

plot_decision_boundary(p_nand, X, Y_NAND, "NAND Decision Boundary")
