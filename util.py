def pearson_correlation_similarity(seq_1, seq_2):
    """

    :param seq_1: type should be np array
    :param seq_2: type should be np array
    :return: pearson_correlation
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

def transform_func_for_X(serial_X, inverse=False):
    """
    This function is to revert the actual X of Depth through serial_X
    Define the process of reverting the reverse direction and identify with R as suffix
    All the time values are after compression

    :param serial_X: a serial which is prepared for reverting
               type: should be np array
    :return: a serial after reverting
    """
    delta = 0.041292
    stable_point = 0.234011

    # values are after delta
    refer_point_Depth = 0.21141
    refer_point_Altim = 0.174779
    ratio = abs(stable_point-refer_point_Altim) / (stable_point-refer_point_Depth)
    if not inverse:
        return stable_point + (serial_X-stable_point) / ratio + delta
    else:
        return (serial_X-delta-stable_point) * ratio + stable_point

def calculate_extremum(X,y):
    extremum_points = []
    point_tuple_ppre = (*X[0], y[0])
    point_tuple_pre = (*X[1], y[1])
    for (comm_xx, yy) in zip(X[2:], y[2:]):
        if (point_tuple_pre[1] > point_tuple_ppre[1]) and \
                (point_tuple_pre[1] > yy):
            extremum_points.append(point_tuple_pre+('maximum',))
        elif (point_tuple_pre[1] < point_tuple_ppre[1]) and \
                (point_tuple_pre[1] < yy):
            extremum_points.append(point_tuple_pre+('minimum',))
        point_tuple_ppre = point_tuple_pre
        point_tuple_pre = (*comm_xx, yy)
    return extremum_points

def calculate_cross_zero(X, y):

    # first element is x(time)
    # second element is y(value)
    # third element is abs() which is the distance between y value and 0
    # fourth element is identification of 'minimum' or 'maximum'
    # fifth element is the abs(slope) at the point crosses zero
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


if __name__ == '__main__':

    # test the function
    transform_func_for_X([])