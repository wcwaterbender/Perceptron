import math

import data

def dot_kf(u, v):
    """
    The basic dot product kernel returns u*v+1.

    Args:
        u: list of numbers
        v: list of numbers

    Returns:
        dot(u,v) + 1
    """
    # TODO
    return 0

def poly_kernel(d):
    """
    The polynomial kernel.

    Args:
        d: a number

    Returns:
        A function that takes two vectors u and v,
        and returns (u*v+1)^d.
    """
    def kf(u, v):
        # TODO: implement the kernel function
        return 0
    return kf

def exp_kernel(s):
    """
    The exponential kernel.

    Args:
        s: a number

    Returns:
        A function that takes two vectors u and v,
        and returns exp(-||u-v||/(2*s^2))
    """
    def kf(u, v):
        # TODO: implement the kernel function
        return 0
    return kf

class Perceptron(object):

    def __init__(self, kf):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.kf = kf
        # TODO: add more fields as needed

    def update(self, point, label):
        """
        Updates the parameters of the perceptron, given a point and a label.

        Args:
            point: a list of numbers
            label: either 1 or -1

        Returns:
            True if there is an update (prediction is wrong),
            False otherwise (prediction is accurate).
        """
        # TODO
        return False

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        return 1

# Feel free to add any helper functions as needed.


if __name__ == '__main__':
    val_data, val_labs = data.load_data('data/validation.csv')
    test_data, test_labs = data.load_data('data/test.csv')
    # TODO: implement code for running the problems
