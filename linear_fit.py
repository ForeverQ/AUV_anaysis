import numpy as np
from scipy.optimize import leastsq


def make_x_y_same_length(Xi, Yi):
    min_len = min(len(Xi), len(Yi))
    return Xi[:min_len], Yi[:min_len]


# function of fitting
def func(p, x):
    k, b = p
    return k * x + b

# deviation
def residuals(p, x, y):
    return y - func(p, x)

# startup value
p0 = [1, 0]

def my_leastsq(x_values, y_values):
    Xi = np.array(x_values)
    Yi = np.array(y_values)
    Xi, Yi =make_x_y_same_length(Xi, Yi)

    Para = leastsq(residuals, p0, args=(Xi, Yi))
    k, b = Para[0]
    return k, b

def my_visualize(x_name, x_values, y_name, y_values, k, b, count):
    import matplotlib.pyplot as plt

    Xi = np.array(x_values)
    Yi = np.array(y_values)
    Xi, Yi =make_x_y_same_length(Xi, Yi)

    plt.figure(figsize=(12,9))
    plt.scatter(Xi,Yi,color="red",label="Sample Point",linewidth=1) # draw samples
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    x = Xi
    y = k * x + b
    plt.plot(x,y,color="orange",label="Fitting Line",linewidth=2) # draw the curve
    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig('linear_picture' + str(count) + '.png', dpi=300)

# parameter of func is y = k * x + b
def get_r_squared2(func, x_values, y_values):

    # TODO to be improved
    Xi = np.array(x_values)
    Yi = np.array(y_values)
    Xi, Yi =make_x_y_same_length(Xi, Yi)

    y_bar = np.sum(Yi) / len(Yi)
    ss_t = np.sum(list(map(lambda x,y: (x-y)**2,Yi,[y_bar]*len(Yi))))
    Y = np.array(list(map(func,Xi)))
    ss_r = np.sum(list(map(lambda x,y: (x-y)**2,Y,[y_bar]*len(Yi))))
    return ss_r / ss_t