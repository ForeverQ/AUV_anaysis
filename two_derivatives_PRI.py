import numpy as np
from my_regression import PolynomialRegressionInterval
import pre_process as p_p
import linear_regression as l_r
import util as t_l
import polynomial_regression_interval as p_r_i


DEGREE_NOW = 6
DISPLAYED_CROSS_ZERO_NUM_1=10
DISPLAYED_CROSS_ZERO_NUM_2=10
NEAR_DISTANCE=0.01


def draw_d_and_d_interval(key_name_1, key_name_2, X_1, X_2, reg_func_d_interval_1, reg_func_d_interval_2,
                          degree):
    """Displays comparison of derivatives and their cross-zero points of two data values.
       Especially, paints the cross-zero points which are close to each other to
       visualize the possibility of correlation between derivatives

    Args:
        key_name_1: name of the first value
        key_name_2: name of the second value
        X_1: X of the first value
        X_2: X of the second value
        reg_func_interval_1: functions list of the first value
        reg_func_interval_2: functions list of the second value
        degree: degree of the function now

    Returns: No return
    """

    # Finds comm_x for picture
    comm_left = X_1[0] if X_1[0]>X_2[0] else X_2[0]
    comm_right = X_1[-1] if X_1[-1]<X_2[-1] else X_2[-1]
    comm_X = np.linspace(comm_left,comm_right,5000)
    comm_X = comm_X.reshape(comm_X.shape[0], 1)
    comm_X_quad = t_l.generate_polynomials(comm_X, degree)
    comm_X_quad_d = t_l.generate_polynomials(comm_X, degree-1)

    # Uses cross-zero points to find extremes of functions
    dp_num_1 = DISPLAYED_CROSS_ZERO_NUM_1
    y_d_dp_1 = reg_func_d_interval_1.predict_d(comm_X_quad_d)
    cross_zero_points_1 = t_l.calculate_cross_zero(comm_X, y_d_dp_1)
    cross_zero_points_1_sorted = sorted(cross_zero_points_1, key=lambda a:a[4], reverse=True)
    top_X_cross_zero_list_1 = [*map(lambda a:a[0],
                                    cross_zero_points_1_sorted[:dp_num_1] \
                                        if len(cross_zero_points_1_sorted)>dp_num_1 \
                                        else cross_zero_points_1_sorted)]
    dp_num_2 = DISPLAYED_CROSS_ZERO_NUM_2
    y_d_dp_2 = reg_func_d_interval_2.predict_d(comm_X_quad_d)
    cross_zero_points_2 = t_l.calculate_cross_zero(comm_X, y_d_dp_2)
    cross_zero_points_2_sorted = sorted(cross_zero_points_2, key=lambda a:a[4], reverse=True)
    top_X_cross_zero_list_2 = [*map(lambda a:a[0],
                                    cross_zero_points_2_sorted[:dp_num_2] \
                                        if len(cross_zero_points_2_sorted)>dp_num_2 \
                                        else cross_zero_points_2_sorted)]
    total_near_count = 0
    for index_2, item_2 in enumerate(top_X_cross_zero_list_2):
        for index_1, item_1 in enumerate(top_X_cross_zero_list_1):
            if abs(item_1-item_2) <= NEAR_DISTANCE:
                if cross_zero_points_2_sorted[index_2][-1] != 'catch':
                    cross_zero_points_2_sorted[index_2] += ('catch',)
                    total_near_count += 1
                if cross_zero_points_1_sorted[index_1][-1] != 'catch':
                    cross_zero_points_1_sorted[index_1] += ('catch',)
                    total_near_count += 1
    # For debug
    # total_czp_num = len(top_X_cross_zero_list_2) + len(top_X_cross_zero_list_1)
    # print(total_near_count, total_near_count/total_czp_num)
    # print(key_name_1,key_name_2)
    # print()

    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, sharex=True, figsize=(16,9))

    # Plot 1
    axarr[0].set_title('derivative 1')
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel(key_name_1)
    axarr[0].grid(True)
    y_d_1_range = np.amax(y_d_dp_1) - np.amin(y_d_dp_1)
    picture_y_d_1_max = np.amax(y_d_dp_1)+0.08*y_d_1_range
    picture_y_d_1_min = np.amin(y_d_dp_1)-0.08*y_d_1_range
    axarr[0].axis([comm_left,comm_right,picture_y_d_1_min,picture_y_d_1_max])
    axarr[0].plot(comm_X, y_d_dp_1, color='c', linewidth=3)
    axarr[0].plot(comm_X, [0]*len(comm_X), color='black', linewidth=1)

    # Draws cross-zero data points
    has_draw_maximum_1 = False
    has_draw_minimum_1 = False
    for index, item in enumerate(cross_zero_points_1_sorted[:dp_num_1] \
                             if len(cross_zero_points_1_sorted)>dp_num_1 else cross_zero_points_1_sorted):
        if item[3]=='maximum':
            if item[-1] == 'catch':
                color_now = 'red'
            else:
                color_now = 'blue'
            axarr[0].scatter([item[0],],[item[1],], 30, color=color_now)
            axarr[0].annotate('%.3f' % item[0],color=color_now,
                              xy=(item[0], item[1]), xycoords='data',
                              xytext=(+10, +10), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color_now))
            if not has_draw_maximum_1:
                axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color=color_now, label='cz_maximum', linewidth=1.5, linestyle="--")
            else:
                axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color=color_now, linewidth=1.5, linestyle="--")
            has_draw_maximum_1 = True
        elif item[3]=='minimum':
            if item[-1] == 'catch':
                color_now = 'red'
            else:
                color_now = 'g'
            axarr[0].scatter([item[0],],[item[1],], 30, color=color_now)
            axarr[0].annotate('%.3f' % item[0],color=color_now,
                              xy=(item[0], item[1]), xycoords='data',
                              xytext=(+10, +10), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color_now))
            if not has_draw_minimum_1:
                axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color=color_now, label='cz_minimum', linewidth=1.5, linestyle="--")
            else:
                axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color=color_now, linewidth=1.5, linestyle="--")
            has_draw_minimum_1 = True
    axarr[0].legend(loc='lower right')

    # Plot 2
    axarr[1].set_title('derivative 2')
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel(key_name_2)
    axarr[1].grid(True)
    y_d_2_range = np.amax(y_d_dp_2) - np.amin(y_d_dp_2)
    picture_y_d_2_max = np.amax(y_d_dp_2)+0.08*y_d_2_range
    picture_y_d_2_min = np.amin(y_d_dp_2)-0.08*y_d_2_range
    axarr[1].axis([comm_left,comm_right,picture_y_d_2_min,picture_y_d_2_max])
    axarr[1].plot(comm_X, y_d_dp_2, color='c', linewidth=3)
    axarr[1].plot(comm_X, [0]*len(comm_X), color='black', linewidth=1)
    has_draw_maximum_2 = False
    has_draw_minimum_2 = False
    for index, item in enumerate(cross_zero_points_2_sorted[:dp_num_2] \
                                 if len(cross_zero_points_2_sorted)>dp_num_2 else cross_zero_points_2_sorted):
        if item[3]=='maximum':
            if item[-1] == 'catch':
                color_now = 'red'
            else:
                color_now = 'blue'
            axarr[1].scatter([item[0],],[item[1],], 30, color=color_now)
            axarr[1].annotate('%.3f' % item[0],color=color_now,
                              xy=(item[0], item[1]), xycoords='data',
                              xytext=(+10, +10), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color_now))
            if not has_draw_maximum_2:
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color=color_now, label='cz_maximum', linewidth=1.5, linestyle="--")
            else:
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color=color_now, linewidth=1.5, linestyle="--")
            has_draw_maximum_2 = True
        elif item[3]=='minimum':
            if item[-1] == 'catch':
                color_now = 'red'
            else:
                color_now = 'g'
            axarr[1].scatter([item[0],],[item[1],], 30, color=color_now)
            axarr[1].annotate('%.3f' % item[0],color=color_now,
                              xy=(item[0], item[1]), xycoords='data',
                              xytext=(+10, +10), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=color_now))
            if not has_draw_minimum_2:
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color=color_now, label='cz_minimum', linewidth=1.5, linestyle="--")
            else:
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color=color_now, linewidth=1.5, linestyle="--")
            has_draw_minimum_2 = True
    axarr[1].legend(loc='lower right')
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
                draw_d_and_d_interval(key_name_1, key_name_2, X_1, X_2,
                                      reg_func_d_interval_1=plf_interval_1,
                                      reg_func_d_interval_2=plf_interval_2,
                                      degree=DEGREE_NOW)
                count += 1
    print(count)