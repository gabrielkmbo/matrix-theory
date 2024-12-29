import numpy as np

def least_squares(A, b, param):
        # b = b.reshape(-1, 1)

        n = A.shape[1]

        # make [A, sqrt(param)I] and [b, 0]
        A_prime = np.concatenate((A, np.sqrt(param) * np.eye(n)), axis=0)
        b_prime = np.concatenate((b, np.zeros(n)), axis=0)

        # use the least squares solver
        return np.linalg.lstsq(A_prime, b_prime, rcond=None)[0]

def main():
        A = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([1, 2, 3])
        param = 0.1

        actual_solution = np.linalg.inv(A.T @ A + param * np.eye(A.shape[1])) @ A.T @ b

        print("Least Squares Solution: ")
        print(least_squares(A, b, param))
        print("Actual Solution: ")
        print(actual_solution)

if __name__ == "__main__":
        main()




