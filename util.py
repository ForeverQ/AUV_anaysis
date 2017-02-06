import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def pearson_correlation_similarity(seq_1, seq_2):
    """Calculates pearson correlation similarity from two sequences

    Args:
        seq_1: first sequence of values
        seq_2: second sequence of values

    Return:
        A float value of pearson correlation similarity calculated by the formula
    """
    length = len(seq_1)
    sum_seq_1 = sum(seq_1)
    sum_seq_2 = sum(seq_2)
    square_sum_seq_1 = sum(seq_1 ** 2)
    square_sum_seq_2 = sum(seq_2 ** 2)

    product = sum(seq_1 * seq_2)

    numerator = product - (sum_seq_1 * sum_seq_2 / length)
    denominator = ((square_sum_seq_1 - sum_seq_1 ** 2 / length) * (square_sum_seq_2 - sum_seq_2 ** 2 / length)) ** 0.5

    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def calculate_extreme(X, y):
    """Calculates and returns a list of extreme points"""
    extreme_points = []
    point_tuple_ppre = (*X[0], y[0])
    point_tuple_pre = (*X[1], y[1])
    for (comm_xx, yy) in zip(X[2:], y[2:]):
        if (point_tuple_pre[1] > point_tuple_ppre[1]) and \
                (point_tuple_pre[1] > yy):
            extreme_points.append(point_tuple_pre+('maximum',))
        elif (point_tuple_pre[1] < point_tuple_ppre[1]) and \
                (point_tuple_pre[1] < yy):
            extreme_points.append(point_tuple_pre+('minimum',))
        point_tuple_ppre = point_tuple_pre
        point_tuple_pre = (*comm_xx, yy)
    return extreme_points

def calculate_cross_zero(X, y):
    """Calculates and returns a list of cross-zero points

    Args:
        X: X value
        y: y value

    Returns:
        A list of tuples and each tuple represents a cross zero point

        Tuple details:
            First element is x(time)
            Second element is y(value)
            Third element is abs(distance) which is the distance between y value and 0
            Fourth element is identification of 'minimum' or 'maximum'
            Fifth element is the abs(slope) at the point crosses zero
    """
    cross_zero_points = []
    point_delt_tuple_ppre = (*X[0], y[0], abs(y[0] - 0.0))
    point_delt_tuple_pre = (*X[1], y[1], abs(y[1] - 0.0))
    for (comm_xx, yy) in zip(X[2:], y[2:]):
        delt_now = abs(yy-0.0)
        if (point_delt_tuple_pre[2]<point_delt_tuple_ppre[2]) and \
                (point_delt_tuple_pre[2]<delt_now) and \
                (point_delt_tuple_ppre[1]*yy<0):
            if (point_delt_tuple_ppre[1]<0 and yy>0):
                temp_tuple = point_delt_tuple_pre+('minimum',)
            elif (point_delt_tuple_ppre[1]>0 and yy<0):
                temp_tuple = point_delt_tuple_pre+('maximum',)
            slope_abs = float(abs((yy-point_delt_tuple_ppre[1]) / (comm_xx-point_delt_tuple_ppre[0])))
            cross_zero_points.append(temp_tuple+(slope_abs,))
        point_delt_tuple_ppre = point_delt_tuple_pre
        point_delt_tuple_pre = (*comm_xx, yy, delt_now)
    return cross_zero_points


def calculate_ratio(a_standard_tv):
    """Calculates the compression ratio of timestamp by the range of timestamp"""
    time_delt = a_standard_tv[-1][0] - a_standard_tv[0][0]
    delt_string_list = list(str(int(time_delt)))
    if delt_string_list[0]=='1':
        return len(delt_string_list) - 1
    else:
        return len(delt_string_list)


def my_derivative_poly(coef_):
    """Calculates derivatives of polynomial functions"""
    coef = []
    for i in range(1, len(coef_[0])):
        coef.append(i*coef_[0][i])
    return np.array(coef)


def generate_polynomials(X, degree):
    """generates n+1 polynomial elements for n degree"""
    quad_feature = PolynomialFeatures(degree=degree)
    X_quad = quad_feature.fit_transform(X)
    return X_quad