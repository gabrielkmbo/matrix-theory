import numpy as np

def base_conversion_matrix(V, W, A):
    W_inv = np.linalg.inv(W)
    return np.matmul(W_inv, np.matmul(A, V))

def main():
    V = np.array([[1, 0, 1],
                  [1, 1, 0],
                  [0, 1, 1]])
    W = np.array([[2, 3],
                  [1, 1]])
    A = np.array([[2, 3, 4],
                  [8, 5, 1]])
    print(base_conversion_matrix(V, W, A))

if __name__ == '__main__':
    main()


