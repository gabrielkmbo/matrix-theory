import numpy as np

def pd_power_recursion(A, x, tol=1e-9, max_iters=100, current_iter=0):
    if current_iter >= max_iters:
        return x

    x_next = A @ x
    x_next /= np.linalg.norm(x_next, 2)

    if np.linalg.norm(x - x_next, 2) < tol:
        return x_next

    return pd_power_recursion(A, x_next, tol, max_iters, current_iter + 1)

def main():
    np.random.seed(42)
    n = 4
    B = np.random.rand(n, n)
    A = B @ B.T

    x0 = np.random.rand(n)
    x0 /= np.linalg.norm(x0, 2)

    res = pd_power_recursion(A, x0)

    eigenvalues, eigenvectors = np.linalg.eig(A)
    largest_eigenvalue_index = np.argmax(eigenvalues)
    actual_dominant_eigenvector = eigenvectors[:, largest_eigenvalue_index]

    # Output the results
    print("Approximated Dominant Eigenvector (Recursive Power Iteration):")
    print(res)
    print("\nActual Dominant Eigenvector (NumPy):")
    print(actual_dominant_eigenvector)

if __name__ == "__main__":
    main()


