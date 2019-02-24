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
    s = 1
    for i in range(0,len(u)):
        s += u[i]*v[i]
    return s

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
        return dot_kf(u,v)**d
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
        norm = 0
        for i in range(len(u)):
            norm += (u[i]-v[i])**2
        norm = math.sqrt(norm)
        return math.exp(-(norm)/float(2*s**2))
    return kf

class Perceptron(object):

    def __init__(self, kf, data, label):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.kf = kf
        # TODO: add more fields as needed
        self.data = data
        self.label = label
        self.Mistakesx = []
        self.Mistakesy = []

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
        if(self.predict(point)==label):
            return False
        else:
            return True

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        acc = 0
        for i in range(len(self.Mistakesx)):
            acc += self.Mistakesy[i]*self.kf(self.Mistakesx[i],point)
        return sign(acc)

# Feel free to add any helper functions as needed.
def sign(x):
    if x > 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    val_data, val_labs = data.load_data('data/validation.csv')
    test_data, test_labs = data.load_data('data/test.csv')
    # TODO: implement code for running the problems
    
    def simpleKernel():
        train = Perceptron(dot_kf,val_data,val_labs)
        L = []
        loss = 0
        for i in range(0,len(val_data)):
            if train.update(val_data[i],val_labs[i]): #if theres an update
                train.Mistakesx.append(val_data[i])
                train.Mistakesy.append(val_labs[i])
                loss = loss + 1
            if ((i+1)%100 ==0):
                L.append(float(loss)/(i+1))
        print L

    def polyKernel():
        
        d = [1, 3, 5, 7, 10, 15, 20]
        L = []
        loss = 0
        for i in range(len(d)):
            train = Perceptron(poly_kernel(d[i]),val_data,val_labs)
            for i in range(0,len(val_data)):
                if train.update(val_data[i],val_labs[i]): #if theres an update
                    train.Mistakesx.append(val_data[i])
                    train.Mistakesy.append(val_labs[i])
                    loss = loss + 1
            L.append(float(loss)/1000)
            loss = 0
        print L

    def expKernel(d):
        train = Perceptron(exp_kernel(d),test_data,test_labs)
        L = []
        loss = 0
        for i in range(0,len(test_data)):
            if train.update(test_data[i],test_labs[i]): #if theres an update
                train.Mistakesx.append( val_data[i])
                train.Mistakesy.append(val_labs[i])
                loss = loss + 1
            if ((i+1)%100 ==0):
                L.append(float(loss)/(i+1))
        print L

    simpleKernel()
    polyKernel()
    expKernel(5)
    expKernel(10)