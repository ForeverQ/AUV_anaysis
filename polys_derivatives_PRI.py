import numpy as np
from my_regression import PolynomialRegressionInterval
import pre_process as p_p
import linear_regression as l_r
import util as t_l
import polynomial_regression_interval as p_r_i


DEGREE_NOW = 6
PEARSON_THRESHOLD = 0.91


def draw_two_p_and_d_interval_commX(key_name_1, key_name_2, X_1, X_2, y_1, y_2,
                              reg_func_interval_1, reg_func_interval_2, degree, count):
    """Uses a threshold of Pearson's r to filter correlated data pairs
       Displays both their polynomial functions and derivative functions

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

    # Finds comm_x for one comparison
    comm_left = X_1[0] if X_1[0]>X_2[0] else X_2[0]
    comm_right = X_1[-1] if X_1[-1]<X_2[-1] else X_2[-1]
    comm_X = np.linspace(comm_left,comm_right,3000)
    comm_X = comm_X.reshape(comm_X.shape[0], 1)
    comm_X_quad = t_l.generate_polynomials(comm_X, degree)
    comm_X_quad_d = t_l.generate_polynomials(comm_X, degree-1)
    y_1_func = reg_func_interval_1.predict(comm_X_quad)
    y_2_func = reg_func_interval_2.predict(comm_X_quad)
    y_d_dp_1 = reg_func_interval_1.predict_d(comm_X_quad_d)
    y_d_dp_2 = reg_func_interval_2.predict_d(comm_X_quad_d)

    # Displays pearson
    y_1_2_pearson_c_s = t_l.pearson_correlation_similarity(y_1_func, y_2_func)
    y_d_1_2_pearson_c_s = t_l.pearson_correlation_similarity(y_d_dp_1, y_d_dp_2)
    if abs(y_1_2_pearson_c_s) >= PEARSON_THRESHOLD:
        print(key_name_1, key_name_2)
        print('Pearson of y:', y_1_2_pearson_c_s)
        print('Pearson of y_d:', y_d_1_2_pearson_c_s)
        print()
    else:
        return 1

    # Drawing part
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(16,9))
    fig.suptitle(key_name_1[:-1-key_name_1[::-1].find('.')] + '  && ' +
                 key_name_2[:-1-key_name_2[::-1].find('.')])
    plt.xlabel('time')
    axes[0].set_ylabel(key_name_1[key_name_1.find('.')+1:key_name_1.rfind('.')])
    axes[0].grid(True)
    axes[0].plot(comm_X, y_1_func, color='orange', label='interval regression', linewidth=3)
    axes[0].legend(loc='upper right')
    axes[1].set_ylabel(key_name_2[key_name_2.find('.')+1:key_name_2.rfind('.')])
    axes[1].grid(True)
    axes[1].plot(comm_X, y_2_func, color='orange', label='interval regression', linewidth=3)
    axes[1].legend(loc='upper right')
    axes[2].set_ylabel('Derivative 1st.')
    axes[2].grid(True)
    axes[2].plot(comm_X, y_d_dp_1, color='c', label='derivatives', linewidth=3)
    axes[2].plot(comm_X, [0]*len(comm_X), color='black', linewidth=1)
    axes[2].legend(loc='upper right')
    axes[3].set_ylabel('derivative 2nd.')
    axes[3].grid(True)
    axes[3].plot(comm_X, y_d_dp_2, color='c', label='derivatives', linewidth=3)
    axes[3].plot(comm_X, [0]*len(comm_X), color='black', linewidth=1)
    axes[3].legend(loc='upper right')
    # plt.savefig('candidate_pairs_' + str(count) + '.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    dict_total = p_p.make_total_dict()
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

    # Makes data pairs and conducts comparison
    count = 0
    for index_2, key_name_2 in  enumerate(key_names_ordered_notStable):
        for key_name_1 in key_names_ordered_notStable[index_2:]:
            # if (key_name_1.endswith('Depth.csv') or key_name_1.endswith('Distance.csv')) and \
            #         (key_name_2.endswith('Depth.csv') or key_name_2.endswith('Distance.csv')) and \
            #         (key_name_1 != key_name_2):
            if (key_name_1 != key_name_2):
                X_1, y_1, plf_interval_1 = dict_funcs[key_name_1]
                X_2, y_2, plf_interval_2 = dict_funcs[key_name_2]
                if draw_two_p_and_d_interval_commX(key_name_1, key_name_2, X_1, X_2,
                                                y_1, y_2, plf_interval_1, plf_interval_2,
                                                DEGREE_NOW, count) != 1:
                    count += 1
    print(count)