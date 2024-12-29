import pandas as pd
import numpy as np

def find_singular_values(data):
    U, S, VT = np.linalg.svd(data, full_matrices=False)
    return S

def main():
    file_path = 'personality20_labeled_oriented.csv'
    data = pd.read_csv(file_path)
    print("Singular values: ")
    print(find_singular_values(data))

if __name__ == "__main__":
    main()


