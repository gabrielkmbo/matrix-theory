import numpy as np
import lu_factorization as lu

def solveLxb(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i] - np.dot(L[i, :i], x[:i])
        if L[i, i] == 0:
            return x
        x[i] /= L[i, i]
    return x

def solveUxb(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(U[i, i + 1:], x[i + 1:])
        if U[i, i] == 0:
            return x
        x[i] /= U[i, i]
    return x

def solveAxb(A, b):
    L, U, P = lu.findLU(A)
    b = P.T @ b
    y = solveLxb(L, b)
    x = solveUxb(U, y)
    return x
def main():
    L = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    b = np.array([1, 2, 3])
    x = solveLxb(L, b)
    print(x)

    A = np.array([[2, 4, 2, 0], [1, 1, -1, -2], [-2, -2, 3, 4], [3, 7, 5, 2]])
    print(solveAxb(A, np.array([1, 2, 3, 4])))

    A = np.loadtxt("Data for PSET2/a.csv", delimiter=",")
    b = np.loadtxt("Data for PSET2/b.csv", delimiter=",")
    x = solveAxb(A, b)
    print(x)

if __name__ == "__main__":
    main()