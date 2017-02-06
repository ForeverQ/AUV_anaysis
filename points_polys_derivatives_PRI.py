import numpy as np
from my_regression import PolynomialRegressionInterval
import pre_process as p_p
import linear_regression as l_r
import util as t_l
import polynomial_regression_interval as p_r_i


DEGREE_NOW = 6
DISPLAYED_CROSS_ZERO_NUM_1=10
DISPLAYED_CROSS_ZERO_NUM_2=10


def draw_two_p_and_d_interval(key_name_1, key_name_2, X_1, X_2, y_1, y_2,
                              reg_func_interval_1, reg_func_interval_2, count,
                              dp_num_1, dp_num_2, degree):
    """Displays data points, polynomials, derivatives and cross-zero points of derivatives of two data values

    Args:
        key_name_1: name of the first value
        key_name_2: name of the second value
        X_1: X of the first value
        X_2: X of the second value
        y_1: y of the first value
        y_2: y of the second value
        reg_func_interval_1: functions list of the first value
        reg_func_interval_2: functions list of the second value
        count: the order of data pairs used for exportation
        dp_num_1: number of displayed cross_zero_points of the first value
        dp_num_2: number of displayed cross_zero_points of the second value
        degree: degree of the function now

    Returns: No return
    """
    def draw_cross_zero_lines(plt, cross_zero_points, y_max, y_min):
        """Displays cross-zero points by drawing vertical lines crossing cross-zero points"""
        has_draw_maximum = False
        has_draw_minimum = False
        for index, item in enumerate(cross_zero_points):
            if item[3]=='maximum':
                color_now = 'blue'
                plt.scatter([item[0],],[item[1],], 30, color=color_now)
                plt.annotate('%.3f' % item[0],color=color_now,
                                  xy=(item[0], item[1]), xycoords='data',
                                  xytext=(+10, +10), textcoords='offset points', fontsize=10,
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color_now))
                if not has_draw_maximum:
                    plt.plot([item[0],item[0]],[y_max,y_min], color=color_now, label='cz_maximum', linewidth=1.5, linestyle="--")
                else:
                    plt.plot([item[0],item[0]],[y_max,y_min], color=color_now, linewidth=1.5, linestyle="--")
                has_draw_maximum = True
            elif item[3]=='minimum':
                color_now = 'g'
                plt.scatter([item[0],],[item[1],], 30, color=color_now)
                plt.annotate('%.3f' % item[0],color=color_now,
                                  xy=(item[0], item[1]), xycoords='data',
                                  xytext=(+10, +10), textcoords='offset points', fontsize=10,
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color_now))
                if not has_draw_minimum:
                    plt.plot([item[0],item[0]],[y_max,y_min], color=color_now, label='cz_minimum', linewidth=1.5, linestyle="--")
                else:
                    plt.plot([item[0],item[0]],[y_max,y_min], color=color_now, linewidth=1.5, linestyle="--")
                has_draw_minimum = True
        plt.legend(loc='lower right')

    X_quad_1 = t_l.generate_polynomials(X_1, degree)
    X_1_left, X_1_right = (X_1[0], X_1[-1])
    X_quad_1_d = t_l.generate_polynomials(X_1, degree-1)
    X_quad_2 = t_l.generate_polynomials(X_2, degree)
    X_2_left, X_2_right = (X_2[0], X_2[-1])
    X_quad_2_d = t_l.generate_polynomials(X_2, degree-1)
    y_1_func = reg_func_interval_1.predict(X_quad_1)
    y_1_range = np.amax(y_1_func) - np.amin(y_1_func)
    y_1_max, y_1_min = (np.amax(y_1_func)+0.08*y_1_range, np.amin(y_1_func)-0.08*y_1_range)
    y_2_func = reg_func_interval_2.predict(X_quad_2)
    y_2_range = np.amax(y_2_func) - np.amin(y_2_func)
    y_2_max, y_2_min = (np.amax(y_2_func)+0.08*y_2_range, np.amin(y_2_func)-0.08*y_2_range)

    # Uses cross-zero points to find extremes of functions
    y_d_dp_1 = reg_func_interval_1.predict_d(X_quad_1_d)
    y_1_d_range = np.amax(y_d_dp_1) - np.amin(y_d_dp_1)
    y_1_d_max, y_1_d_min = (np.amax(y_d_dp_1)+0.08*y_1_d_range, np.amin(y_d_dp_1)-0.08*y_1_d_range)
    cross_zero_points_1 = t_l.calculate_cross_zero(X_1, y_d_dp_1)
    cross_zero_points_1_sorted = sorted(cross_zero_points_1, key=lambda a:a[4], reverse=True)
    y_d_dp_2 = reg_func_interval_2.predict_d(X_quad_2_d)
    y_2_d_range = np.amax(y_d_dp_2) - np.amin(y_d_dp_2)
    y_2_d_max, y_2_d_min = (np.amax(y_d_dp_2)+0.08*y_2_d_range, np.amin(y_d_dp_2)-0.08*y_2_d_range)
    cross_zero_points_2 = t_l.calculate_cross_zero(X_2, y_d_dp_2)
    cross_zero_points_2_sorted = sorted(cross_zero_points_2, key=lambda a:a[4], reverse=True)

    # Drawing part
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(18,10))
    fig.suptitle(key_name_1[:-1-key_name_1[::-1].find('.')] + '  && ' +
                 key_name_2[:-1-key_name_2[::-1].find('.')])
    axes[0].scatter(X_1, y_1, color='red', label='Sample Point', linewidths=1)
    axes[0].set_ylabel(key_name_1[:key_name_1.find('.')])
    axes[0].grid(True)
    axes[0].axis([X_1_left,X_1_right,y_1_min,y_1_max])
    axes[0].plot(X_1, y_1_func, color='orange', linewidth=3)
    axes[2].set_ylabel('Derivative 1st.')
    axes[2].grid(True)
    axes[2].axis([X_1_left,X_1_right,y_1_d_min,y_1_d_max])
    axes[2].plot(X_1, y_d_dp_1, color='c', linewidth=3)
    axes[2].plot(X_1, [0]*len(X_1), color='black', linewidth=1)

    # Draws cross-zero data points
    dp_cross_zero_points_1 = cross_zero_points_1_sorted[:dp_num_1] if len(cross_zero_points_1_sorted)>dp_num_1 else cross_zero_points_1_sorted
    draw_cross_zero_lines(axes[2], dp_cross_zero_points_1, y_1_d_max, y_1_d_min)
    fig.subplots_adjust(hspace=0.1)
    plt.setp([a.get_xticklabels() for a in fig.axes], visible=True)
    plt.subplot(4, 1, 2)
    plt.scatter(X_2, y_2, color='red', label='Sample Point', linewidths=1)
    plt.ylabel(key_name_2[:key_name_2.find('.')])
    plt.grid(True)
    plt.axis([X_2_left,X_2_right,y_2_min,y_2_max])
    plt.plot(X_2, y_2_func, color='orange', linewidth=3)
    plt.subplot(4, 1, 4)
    plt.ylabel('Derivative 2nd.')
    plt.grid(True)
    plt.axis([X_2_left,X_2_right,y_2_d_min,y_2_d_max])
    plt.plot(X_2, y_d_dp_2, color='c', linewidth=3)
    plt.plot(X_2, [0]*len(X_2), color='black', linewidth=1)
    dp_cross_zero_points_2 = cross_zero_points_2_sorted[:dp_num_2] if len(cross_zero_points_2_sorted)>dp_num_2 else cross_zero_points_2_sorted
    draw_cross_zero_lines(plt, dp_cross_zero_points_2, y_2_d_max, y_2_d_min)
    plt.show()
    # plt.savefig('cross-zero_points_' + str(count) + '.png', dpi=200)


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

    # Analyzes value pairs
    count = 0
    for index_2, key_name_2 in  enumerate(key_names_ordered_notStable):
        for key_name_1 in key_names_ordered_notStable[index_2:]:
            # if (key_name_1.endswith('Depth.csv') or key_name_1.endswith('Distance.csv')) and \
            #         (key_name_2.endswith('Depth.csv') or key_name_2.endswith('Distance.csv')) and \
            #         (key_name_1 != key_name_2):
            if (key_name_1 != key_name_2):
                X_1, y_1, plf_interval_1 = dict_funcs[key_name_1]
                X_2, y_2, plf_interval_2 = dict_funcs[key_name_2]
                draw_two_p_and_d_interval(key_name_1, key_name_2, X_1, X_2,
                                          y_1, y_2, plf_interval_1, plf_interval_2, count,
                                          dp_num_1=DISPLAYED_CROSS_ZERO_NUM_1, dp_num_2=DISPLAYED_CROSS_ZERO_NUM_2,
                                          degree=DEGREE_NOW)
                count += 1
    print(count)