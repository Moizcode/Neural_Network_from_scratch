import numpy as np


def main():
    x1 = np.random.rand(150)
    x2 = np.random.rand(150)
    x3 = np.random.rand(150)
    y = 5 * x1 + 2.4 * x2 + 7.897 * x3 + np.random.rand(150)

    inputs = np.dstack((x1, x2, x3))
    print(inputs)
    print(f"Input shape: {inputs.shape}, target shape: {y.shape}")


if __name__ == "__main__":
    main()
