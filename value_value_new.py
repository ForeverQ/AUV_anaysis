import numpy as np
import pre_process as p_p
import linear_regression as l_r
import polynomial_regression_interval as p_r_i
from my_regression import PolynomialRegressionInterval
import util as t_l

DEGREE_NOW = 6

def draw_two_value_pairs_commX(key_name_1, key_name_2, X_1, X_2, y_1, y_2,
                              reg_func_interval_1, reg_func_interval_2, degree, count):
    """Combines and draws value pairs

    Args:
        key_name_1: name of the first value
        key_name_2: name of the second value
        X_1: X of the first value
        X_2: X of the second value
        y_1: y of the first value
        y_2: y of the second value
        reg_func_interval_1: functions list of the first value
        reg_func_interval_2: functions list of the second value
        degree: degree of the function now
        count: the order of data pairs used for exportation

    Returns: No return
    """
    comm_left = X_1[0] if X_1[0]>X_2[0] else X_2[0]
    comm_right = X_1[-1] if X_1[-1]<X_2[-1] else X_2[-1]
    comm_X = np.linspace(comm_left,comm_right,1000)
    comm_X = comm_X.reshape(comm_X.shape[0], 1)
    comm_X_quad = t_l.generate_polynomials(comm_X, degree)
    y_1_func = reg_func_interval_1.predict(comm_X_quad)
    y_2_func = reg_func_interval_2.predict(comm_X_quad)

    # Displays pearson
    y_1_2_pearson_c_s = t_l.pearson_correlation_similarity(y_1_func, y_2_func)
    print(key_name_1, key_name_2)
    print('Pearson: ', y_1_2_pearson_c_s)
    print()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 9))
    cm = plt.cm.get_cmap('rainbow')
    z_color = np.arange(len(y_1_func))
    plt.title(key_name_1 + '  && ' + key_name_2)
    plt.xlabel(key_name_1)
    plt.ylabel(key_name_2)
    plt.grid(True)
    sc = plt.scatter(y_1_func, y_2_func, c=z_color, s=100, cmap=cm, label='Sample Points', alpha=.6)
    plt.legend()
    plt.colorbar(sc)
    # plt.savefig('Depth&Distance combinations_' + str(count) + '.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    dict_total = p_p.make_total_dict()

    # key_names ordered by different types
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_notStable = [key_name for key_name in key_names_ordered if key_name not in l_r.STABLE_VALUES]
    t_comps_ratio = t_l.calculate_ratio(a_standard_tv=dict_total[' x (m/s/s).Acceleration.csv'])
    dict_funcs = dict()
    for key_name in key_names_ordered_notStable:
        X, X_quad, y = p_r_i.make_plf_cols(dict_total[key_name], key_name, t_comps_ratio,
                                               degree=DEGREE_NOW, GaussFiltered=True)
        plf_interval = PolynomialRegressionInterval()
        plf_interval.fit(X_quad, y)
        plf_interval.calculate_derivatives()
        dict_funcs[key_name] = (X, y, plf_interval)

    # Analyzes value pairs
    count = 0
    for key_name_2 in key_names_ordered_notStable:
        for key_name_1 in key_names_ordered_notStable:
            # if (key_name_1.endswith('Depth.csv') or key_name_1.endswith('Distance.csv')) and \
            #         (key_name_2.endswith('Depth.csv') or key_name_2.endswith('Distance.csv')) and \
            #         (key_name_1 != key_name_2):
            if (key_name_1 != key_name_2):
                X_1, y_1, plf_interval_1 = dict_funcs[key_name_1]
                X_2, y_2, plf_interval_2 = dict_funcs[key_name_2]
                count += 1
                draw_two_value_pairs_commX(key_name_1, key_name_2, X_1, X_2, y_1, y_2,
                                           plf_interval_1, plf_interval_2, DEGREE_NOW, count)