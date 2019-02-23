# Logistic Regression using SGD [100 points]#
(this project is a replica of a test project from Yue Zhang)

First get the assignment by running:
    
    download and unzip Perceptron.zip
    cd Perceptron

You will be modifying the file `perceptron.py`.

In this problem, we seek to perform a digit recognition task, where we are given an image of a handwritten digit and wish to predict what number it represents. This is a special case of an important area of language processing known as Optical Character Recognition (OCR). We will be simplifying our goal to that of a binary classification between two relatively hard-to-distinguish numbers (specifically, predicting a '3' versus a '5'). To do this, you will implement a kernelized version of the Perceptron algorithm.

### Data

The digit images have been taken from the Kaggle competition linked to on the projects page, \url{http://www.kaggle.com/c/digit-recognizer/data}. This data was originally from the MNIST database of handwritten digits, but was converted into a easier-to-use file format.

The data has also undergone some preprocessing. It has been filtered to just those datapoints whose labels are 3 or 5, which have then been relabeled to 1 and -1 respectively. Then, 1000-point samples have been created, named \emph{validation.csv} and \emph{test.csv}. The first column of these files is the label of each point, followed by the grayscale value of each pixel.


### Perceptron [40 points]

First, get the Perceptron working with the kernel $k_p^1(u,v) = u \cdot v + 1$.

Implement the function `dot_kf`, and implement the methods for `Perceptron`.

Run Perceptron for a single pass over the validation set with this kernel, and plot the average loss as a function of the number of steps. Record the average loss every 100 steps, e.g. [100, 200, 300, ...]. The loss is the number of times the perceptron classified incorrectly divided by the total number of points classified so far.

Run the perceptron in the `if __name__ == '__main__':` block at the bottom of the code.

### Polynomial kernel [30 points]

Implement the function `poly_kernel`.

For the set `d = [1,3,5,7,10,15,20]`, run Perceptron for a single pass over the validation set with `poly_kernel(d)`, and plot the average loss over the entire validation set as a function of `d`. What value of `d` produces the lowest loss?

### Exponential kernel [30 points]

Implement the function `exp_kernel`.

For the `d` you chose in the previous step, run Perceptron with both `poly_kernel(d)` and `exp_kernel(10)` for a single pass over the testing set. For each of these two kernels, plot the average loss as a function of the number of steps. As above, report the average loss for every 100 steps.

### Submitting
1. Submit your modified `perceptron.py` file and a PDF file containing all of the plots to the GauchoSpace.
2.Also print out your "perceptron.py" and the PDF file with plots. Attach them to your homework (this is what you should turn in as a hardcopy in class).