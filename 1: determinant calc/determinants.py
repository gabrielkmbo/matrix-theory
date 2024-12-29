import numpy as np

def calc_det(matrix):
    if matrix.shape == (1, 1):
        return matrix[0, 0]

    det = 0

    for col in range(matrix.shape[1]):
        sign = (-1) ** (0 + col)
        new_matrix = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)

        det += sign * matrix[0, col] * calc_det(new_matrix)

    return det

def main():
    three_by_three = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    two_by_two = np.array([[1, 2], [3, 4]])
    six_by_six = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            six_by_six[i][j] = np.random.randint(1, 10)
    print(six_by_six)
    print("Determinant using numpy: ", np.linalg.det(six_by_six))
    print("Determinant using function: ", calc_det(six_by_six))
    # expect: 0
    print(calc_det(three_by_three))
    # expect: -2
    print(calc_det(two_by_two))

if __name__ == '__main__':
    main()