import numpy as np

def complex_invers(F):
    Ft = F.T
    identity = np.identity(F.shape[0])
    identity_10 = np.identity(10)
    inverse = np.linalg.inv((identity_10 + Ft @ F))
    return identity - F @ i qnverse @ Ft


def main():
    F = np.random.rand(10000, 10)
    print(complex_invers(F))


if __name__ == "__main__":
    main()


