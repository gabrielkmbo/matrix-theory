import numpy as np
from least_squares import least_squares
import matplotlib.pyplot as plt

TRAIN_SIZE = 200
# each value of λ of the form λ = 10α, where α are equispaced
# exponents ranging between −5 and 1 (you may take the spacing to be 0.1)
LAMBDAS = [10**i for i in np.arange(-5, 1, 0.1)]

def train(A, b):
    res = []
    for lambda_ in LAMBDAS:
        res.append(least_squares(A, b, lambda_))
    return res

def test(A, b, x):
    prediction_error = []
    for sol in x:
        error = np.linalg.norm(A @ sol - b) ** 2 / 70
        prediction_error.append(error)
    return prediction_error

def predict(A, b):
    # Rescale the columns of A so that they each have unit norm.
    A_norm = A / np.linalg.norm(A, axis=0)

    # Split the data into a training set and a test set.
    train_rows = np.random.choice(len(A), TRAIN_SIZE, replace=False)
    # Pick the 70 rows not chosen in the test set.
    test_rows = list(set(range(0, len(A))) - set(train_rows))

    train_A = A_norm[train_rows]
    train_b = b[train_rows]

    solutions = train(train_A, train_b)
    prediction_error = test(A_norm[test_rows], b[test_rows], solutions)

    # Plot the lambda values against the prediction error.
    plt.plot(LAMBDAS, prediction_error, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('λ')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs λ')
    plt.show()

    print("Minimum Prediction Error: ", min(prediction_error))
    print("Difference to lambda = 0: ", prediction_error[0] - min(prediction_error))

def main():
    A = np.loadtxt("A.csv", delimiter=",")
    b = np.loadtxt("b.csv", delimiter=",")
    predict(A, b)

if __name__ == "__main__":
    main()












