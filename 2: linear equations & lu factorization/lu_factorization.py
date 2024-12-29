import numpy as np

def findLU(matrix, L=None, P=None, level = 0):
    n = matrix.shape[0]

    # Initialize L and P if not provided
    if L is None:
        L = np.eye(n)
    if P is None:
        P = np.eye(n)

    # Base case: when the submatrix is 1x1 or empty, return L, U, P
    if n <= level:
        return L, matrix, P

    # Check if the diagonal element is zero and if the column has a non-zero value
    first_col = matrix[level:, level]
    if matrix[level, level] == 0 and np.any(first_col != 0):
        # Find the first non-zero entry in the current column below the diagonal to swap
        non_zero_idx = level + np.where(first_col != 0)[0][0]
        # Swap rows in the matrix
        matrix[[level, non_zero_idx]] = matrix[[non_zero_idx, level]]
        # Swap rows in the permutation matrix P
        P[[level, non_zero_idx]] = P[[non_zero_idx, level]]
        # Swap rows in L up to the current level
        if level > 0:
            L[[level, non_zero_idx], :level] = L[[non_zero_idx, level], :level]

    # Proceed with the elimination
    for i in range(level + 1, n):
        if matrix[level, level] != 0:  # Avoid division by zero
            factor = matrix[i, level] / matrix[level, level]
            matrix[i, :] = matrix[i, :] - factor * matrix[level, :]  # Eliminate entries below the pivot
            L[i, level] = factor  # Store the factor in the L matrix

    # Recursive call on the submatrix (excluding the current row and column)
    L, U, P = findLU(matrix, L, P, level + 1)

    return L, U, P


def main():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    L, U, P = findLU(matrix)
    print("Permutation Matrix P:\n", P)
    print("Lower Triangular Matrix L:\n", L)
    print("Upper Triangular Matrix U:\n", U)

    # Check if HW prob 1 is correct
    A = np.array([[2, 4, 2, 0], [1, 1, -1, -2], [-2, -2, 3, 4], [3, 7, 5, 2]])
    L, U, P = findLU(A)
    print("Permutation Matrix P:\n", P)
    print("Lower Triangular Matrix L:\n", L)
    print("Upper Triangular Matrix U:\n", U)

    twenty_by_twenty = np.loadtxt("Data for PSET2/a.csv", delimiter=",")
    L, U, P = findLU(twenty_by_twenty)
    det = np.prod(np.diag(U))
    print("Permutation Matrix P:\n", P)
    print("Lower Triangular Matrix L:\n", L)
    print("Upper Triangular Matrix U:\n", U)
    print("Determinant of A:", det)

if __name__ == "__main__":
    main()
