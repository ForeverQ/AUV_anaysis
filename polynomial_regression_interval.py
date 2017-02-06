import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
from scipy.ndimage import filters
import pre_process as p_p
import linear_regression as l_r
import polynomial_regression as p_r
from my_regression import PolynomialRegressionInterval
import util as t_l


def make_plf_cols(tv_list, key_name, t_comps_ratio, degree, GaussFiltered=True, GF_s=8):

    # for testing
    # key_name = ' PayloadPowerSwitch - Channel 7 (Ethernet Switch).Voltage.csv'
    # tv_list = dict_total[key_name]

    pd.set_option('precision',8)
    df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
    df_tv.sort_values(by='time')
    X_col_list = list(df_tv['time'])
    y_col_list = list(df_tv[key_name])

    # let the time value be smaller
    # TODO should have done some change on this(Already Done)
    # X_delta_sp = X_col_list[0] / (10**t_comps_ratio)
    # X_col_list_cms = [*map(lambda a: a/(10**t_comps_ratio)-X_delta_sp, X_col_list)]
    X_delta_sp = X_col_list[0] // (10**t_comps_ratio)
    X_col_list_cms = [*map(lambda a: a/(10**t_comps_ratio)-X_delta_sp, X_col_list)]

    # x_x = np.linspace(list(X_col)[0], list(X_col)[-1],len(X_col))
    X = np.array(X_col_list_cms).reshape(len(X_col_list_cms),1)
    y = np.array(y_col_list).reshape(len(y_col_list),1)
    X_quad = t_l.generate_polynomials(X, degree)
    if GaussFiltered:
        y_list = [item[0] for item in y]
        y_GF = filters.gaussian_filter1d(y_list,GF_s)
        return (X, X_quad, np.array(y_GF))
    else:
        return (X, X_quad, y)

def draw_two_p_and_d_interval_commX(key_name_1, key_name_2, X_1, X_2, y_1, y_2,
                              reg_func_interval_1, reg_func_interval_2, degree, count):
    # find comm_x for picture
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

    # display pearson
    y_1_2_pearson_c_s = t_l.pearson_correlation_similarity(y_1_func, y_2_func)
    y_d_1_2_pearson_c_s = t_l.pearson_correlation_similarity(y_d_dp_1, y_d_dp_2)

    # flexible judge condition for test
    if abs(y_1_2_pearson_c_s)>=0.91:
        print(key_name_1, key_name_2)
        print('y:', y_1_2_pearson_c_s)
        print('y_d:', y_d_1_2_pearson_c_s)
        print()
        # return 0
    else:
        return 1

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(16,9))
    # fig.tight_layout()
    fig.suptitle(key_name_1[:-1-key_name_1[::-1].find('.')] + '  && ' +
                 key_name_2[:-1-key_name_2[::-1].find('.')])
    plt.xlabel('time')
    axes[0].set_ylabel(key_name_1[key_name_1.find('.')+1:key_name_1.rfind('.')])
    axes[0].grid(True)
    # axes[0].axis([X_1_left,X_1_right,y_1_min,y_1_max])
    axes[0].plot(comm_X, y_1_func, color='orange', label='interval regression', linewidth=3)
    axes[0].legend(loc='upper right')
    axes[1].set_ylabel(key_name_2[key_name_2.find('.')+1:key_name_2.rfind('.')])
    axes[1].grid(True)
    # axes[1].axis([X_1_left,X_1_right,y_1_min,y_1_max])
    axes[1].plot(comm_X, y_2_func, color='orange', label='interval regression', linewidth=3)
    axes[1].legend(loc='upper right')
    axes[2].set_ylabel('Derivative 1st.')
    axes[2].grid(True)
    # axes[2].axis([X_1_left,X_1_right,y_1_d_min,y_1_d_max])
    axes[2].plot(comm_X, y_d_dp_1, color='c', label='derivatives', linewidth=3)
    axes[2].plot(comm_X, [0]*len(comm_X), color='black', linewidth=1)
    axes[2].legend(loc='upper right')
    axes[3].set_ylabel('derivative 2nd.')
    axes[3].grid(True)
    # axes[3].axis([X_1_left,X_1_right,y_1_d_min,y_1_d_max])
    axes[3].plot(comm_X, y_d_dp_2, color='c', label='derivatives', linewidth=3)
    axes[3].plot(comm_X, [0]*len(comm_X), color='black', linewidth=1)
    axes[3].legend(loc='upper right')
    # plt.savefig('candidate_pairs_' + str(count) + '.png', dpi=200)
    plt.show()

def draw_points_and_poly_interval(key_name, X, y, reg_func_interval, degree, count):
    import matplotlib.pyplot as plt
    X_quad = t_l.generate_polynomials(X, degree)
    interval_values = [*map(lambda a:a[1], reg_func_interval.interval_values_quad_)]
    plt.figure(figsize=(16,9))
    plt.title(key_name[:-1-key_name[::-1].find('.')] + '\'s Interval Regression')
    plt.xlabel('time')
    plt.ylabel(key_name)
    plt.grid(True)
    plt.scatter(X, y, color='red', label='Sample Point', linewidths=1)
    y_func = reg_func_interval.predict(X_quad)
    picture_y_min = np.amin(y_func)
    picture_y_max = np.amax(y_func)
    plt.plot(X, y_func, color='orange', label='degree ' + str(degree), linewidth=3)

    # TODO to be improved about the last value in interval
    # for x_interval, y_interval in zip(interval_values[:-1], reg_func.predict(reg_func.interval_values_quad_)[:-1]):
    #     plt.scatter([x_interval,], [y_interval,], 30, color ='k')

    for x_interval in interval_values:
        plt.plot([x_interval,x_interval],[picture_y_min,picture_y_max], color='k', linewidth=1.5, linestyle="--")
    plt.legend(loc='upper left')
    # plt.savefig('points_poly_' + str(count) + '.png', dpi=200)
    plt.show()

def draw_d_and_d_interval(key_name_1, key_name_2, X_1, X_2, reg_func_d_interval_1, reg_func_d_interval_2,
                          degree, draw_type='dd'):

    # find comm_x for picture
    comm_left = X_1[0] if X_1[0]>X_2[0] else X_2[0]
    comm_right = X_1[-1] if X_1[-1]<X_2[-1] else X_2[-1]

    # do some change
    comm_X = np.linspace(comm_left,comm_right,5000)
    comm_X = comm_X.reshape(comm_X.shape[0], 1)
    comm_X_quad = t_l.generate_polynomials(comm_X, degree)
    comm_X_quad_d = t_l.generate_polynomials(comm_X, degree-1)

    # find extremum(use the derivative way)
    dp_num_1 = 10
    y_d_dp_1 = reg_func_d_interval_1.predict_d(comm_X_quad_d)
    cross_zero_points_1 = t_l.calculate_cross_zero(comm_X, y_d_dp_1)
    cross_zero_points_1_sorted = sorted(cross_zero_points_1, key=lambda a:a[4], reverse=True)
    top_X_cross_zero_list_1 = [*map(lambda a:a[0],
                                    cross_zero_points_1_sorted[:dp_num_1] \
                                        if len(cross_zero_points_1_sorted)>dp_num_1 \
                                        else cross_zero_points_1_sorted)]
    dp_num_2 = 10
    y_d_dp_2 = reg_func_d_interval_2.predict_d(comm_X_quad_d)
    cross_zero_points_2 = t_l.calculate_cross_zero(comm_X, y_d_dp_2)
    cross_zero_points_2_sorted = sorted(cross_zero_points_2, key=lambda a:a[4], reverse=True)
    top_X_cross_zero_list_2 = [*map(lambda a:a[0],
                                    cross_zero_points_2_sorted[:dp_num_2] \
                                        if len(cross_zero_points_2_sorted)>dp_num_2 \
                                        else cross_zero_points_2_sorted)]
    # TODO pay attention to the relation between 'top_X_cross_zero_list_2' and 'cross_zero_points_2_sorted'
    total_near_count = 0
    for index_2, item_2 in enumerate(top_X_cross_zero_list_2):
        for index_1, item_1 in enumerate(top_X_cross_zero_list_1):
            if abs(item_1-item_2)<=0.01:
                if cross_zero_points_2_sorted[index_2][-1] != 'catch':
                    cross_zero_points_2_sorted[index_2] += ('catch',)
                    total_near_count += 1
                if cross_zero_points_1_sorted[index_1][-1] != 'catch':
                    cross_zero_points_1_sorted[index_1] += ('catch',)
                    total_near_count += 1
    total_czp_num = len(top_X_cross_zero_list_2) + len(top_X_cross_zero_list_1)

    # if (total_czp_num != 0) and (total_near_count / total_czp_num >= 0.75):
    print(total_near_count, total_near_count/total_czp_num)
    print(key_name_1,key_name_2)
    print()
    """
    with open('candidate_correlated_pairs.txt', 'a') as f:
        f.write('percentage: ' + str(total_near_count/total_czp_num) + '\n')
        f.write('candidate pair: ' + key_name_1 + ', ' + key_name_2 + '\n')
        f.write('\n')
    """

    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, sharex=True, figsize=(16,9))
    # fig.tight_layout()

    # This part should be regardless
    if draw_type=='pd':
        X_cross_zero_list_1 = [*map(lambda a:a[0], cross_zero_points_1)]
        X_cross_zero = np.array(X_cross_zero_list_1).reshape(len(X_cross_zero_list_1), 1)
        X_quad_cross_zero = t_l.generate_polynomials(X_cross_zero,degree)
        # TODO cause long time
        y_cross_zero = reg_func_d_interval_1.predict(X_quad_cross_zero)
        extremum_points_by_d = [(item_for_X[0],item_for_y,item_for_X[3],item_for_X[4]) for item_for_X, item_for_y in zip(cross_zero_points_1, y_cross_zero)]
        extremum_points_by_d_sorted = sorted(extremum_points_by_d, key=lambda a:a[3], reverse=True)
        # plot 1
        axarr[0].set_xlabel('time')
        axarr[0].set_ylabel(key_name_1)
        axarr[0].grid(True)
        # TODO cause long time
        y_dp_1 = reg_func_d_interval_1.predict(comm_X_quad)
        y_1_range = np.amax(y_dp_1) - np.amin(y_dp_1)
        picture_y_d_1_max = np.amax(y_dp_1)+0.08*y_1_range
        picture_y_d_1_min = np.amin(y_dp_1)-0.08*y_1_range
        axarr[0].axis([comm_left,comm_right,picture_y_d_1_min,picture_y_d_1_max])
        axarr[0].plot(comm_X, y_dp_1, color='orange', label='degree '+str(degree), linewidth=3)

        """
        # find extremum
        extremum_points = t_l.calculate_extreme(comm_X, y_dp_1)
        for index, item in enumerate(extremum_points):
            if item[2]=='maximum':
                axarr[0].scatter([item[0],],[item[1],], 30, color ='red')
                axarr[0].annotate('%.3f' % item[0],color='red',
                                  xy=(item[0], item[1]), xycoords='data',
                                  xytext=(-30, -30), textcoords='offset points', fontsize=10,
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'))
                if index in (0,1):
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_1_min], color='red', label='maximum', linewidth=1.5, linestyle="--")
                else:
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_1_min], color='red', linewidth=1.5, linestyle="--")
            elif item[2]=='minimum':
                axarr[0].scatter([item[0],],[item[1],], 30, color ='m')
                axarr[0].annotate('%.3f' % item[0],color='m',
                                  xy=(item[0], item[1]), xycoords='data',
                                  xytext=(-30, -30), textcoords='offset points', fontsize=10,
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='m'))
                if index in (0,1):
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_1_min], color='m', label='minimum', linewidth=1.5, linestyle="--")
                else:
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_1_min], color='m', linewidth=1.5, linestyle="--")
        """

        for index, item in enumerate(extremum_points_by_d_sorted[:dp_num_1] \
                                     if len(extremum_points_by_d_sorted)>dp_num_1 else extremum_points_by_d_sorted):
            if item[2]=='maximum':
                axarr[0].scatter([item[0],],[item[1],], 30, color ='red')
                axarr[0].annotate('%.3f' % item[0],color='red',
                                  xy=(item[0], item[1]), xycoords='data',
                                  xytext=(-30, -30), textcoords='offset points', fontsize=10,
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'))
                if index in (0,1):
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color='red', label='maximum', linewidth=1.5, linestyle="--")
                else:
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color='red', linewidth=1.5, linestyle="--")
            elif item[2]=='minimum':
                axarr[0].scatter([item[0],],[item[1],], 30, color ='m')
                axarr[0].annotate('%.3f' % item[0],color='m',
                                  xy=(item[0], item[1]), xycoords='data',
                                  xytext=(-30, -30), textcoords='offset points', fontsize=10,
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='m'))
                if index in (0,1):
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color='m', label='minimum', linewidth=1.5, linestyle="--")
                else:
                    axarr[0].plot([item[0],item[0]],[item[1],picture_y_d_1_min], color='m', linewidth=1.5, linestyle="--")

        axarr[0].legend(loc='upper right')
    elif draw_type=='dd':
        # plot 1
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

        # arrange cross 0 data points
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

    # plot 2
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

    # find cross 0 data points
    has_draw_maximum_2 = False
    has_draw_minimum_2 = False
    # cross_zero_points_2 = t_l.calculate_cross_zero(comm_X, y_d_dp_2)
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

def draw_two_p_and_d_interval(key_name_1, key_name_2, X_1, X_2, y_1, y_2,
                              reg_func_interval_1, reg_func_interval_2, count,
                              dp_num_1, dp_num_2, degree):

    def draw_cross_zero_lines(plt, cross_zero_points, y_max, y_min):
        # draw cross 0 data points
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

    # find extremum(use the derivative way)
    y_d_dp_1 = reg_func_interval_1.predict_d(X_quad_1_d)
    y_1_d_range = np.amax(y_d_dp_1) - np.amin(y_d_dp_1)
    y_1_d_max, y_1_d_min = (np.amax(y_d_dp_1)+0.08*y_1_d_range, np.amin(y_d_dp_1)-0.08*y_1_d_range)
    cross_zero_points_1 = t_l.calculate_cross_zero(X_1, y_d_dp_1)
    cross_zero_points_1_sorted = sorted(cross_zero_points_1, key=lambda a:a[4], reverse=True)
    top_X_cross_zero_list_1 = [*map(lambda a:a[0],
                                    cross_zero_points_1_sorted[:dp_num_1] \
                                        if len(cross_zero_points_1_sorted)>dp_num_1 \
                                        else cross_zero_points_1_sorted)]
    y_d_dp_2 = reg_func_interval_2.predict_d(X_quad_2_d)
    y_2_d_range = np.amax(y_d_dp_2) - np.amin(y_d_dp_2)
    y_2_d_max, y_2_d_min = (np.amax(y_d_dp_2)+0.08*y_2_d_range, np.amin(y_d_dp_2)-0.08*y_2_d_range)
    cross_zero_points_2 = t_l.calculate_cross_zero(X_2, y_d_dp_2)
    cross_zero_points_2_sorted = sorted(cross_zero_points_2, key=lambda a:a[4], reverse=True)
    top_X_cross_zero_list_2 = [*map(lambda a:a[0],
                                    cross_zero_points_2_sorted[:dp_num_2] \
                                        if len(cross_zero_points_2_sorted)>dp_num_2 \
                                        else cross_zero_points_2_sorted)]



    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(18,10))
    # fig.tight_layout()
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



    # draw cross 0 data points
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

    # draw cross 0 data points
    dp_cross_zero_points_2 = cross_zero_points_2_sorted[:dp_num_2] if len(cross_zero_points_2_sorted)>dp_num_2 else cross_zero_points_2_sorted
    draw_cross_zero_lines(plt, dp_cross_zero_points_2, y_2_d_max, y_2_d_min)

    # plt.show()
    plt.savefig('cross-zero_points_' + str(count) + '.png', dpi=200)


if __name__ == '__main__':
    dict_total = p_p.make_total_dict()

    # key_names in order by different types
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_notStable = [key_name for key_name in key_names_ordered if key_name not in l_r.STABLE_VALUES]

    # for testing
    key_names_ordered_DDValues = [key_name for key_name in key_names_ordered if key_name in l_r.DD_VALUES]

    degree_now = 6
    t_comps_ratio = t_l.calculate_ratio(a_standard_tv=dict_total[' x (m/s/s).Acceleration.csv'])
    # calculate a dictionary of key_name and its interval function
    dict_funcs = dict()
    for key_name in key_names_ordered_notStable:
        X, X_quad, y = make_plf_cols(dict_total[key_name], key_name, t_comps_ratio,
                                               degree=degree_now, GaussFiltered=True)
        plf_interval = PolynomialRegressionInterval()
        # for debug
        # print('r2 for ' + key_name)
        plf_interval.fit(X_quad, y)
        plf_interval.calculate_derivatives()
        dict_funcs[key_name] = (X, y, plf_interval)

    # analyze the interval regression
    count = 0
    for key_name in key_names_ordered_notStable:
        if key_name.endswith('Distance.csv') or key_name.endswith('Depth.csv'):
            X, y, plf_interval = dict_funcs[key_name]
            draw_points_and_poly_interval(key_name, X, y, plf_interval, degree_now, count)
            count += 1
    exit(0)

    # analyze value pairs
    count = 0
    for index_2, key_name_2 in  enumerate(key_names_ordered_notStable):
        for key_name_1 in key_names_ordered_notStable[index_2:]:
            if (key_name_1.endswith('Depth.csv') or key_name_1.endswith('Distance.csv')) and \
                    (key_name_2.endswith('Depth.csv') or key_name_2.endswith('Distance.csv')) and \
                    (key_name_1 != key_name_2):
            # if (key_name_1 != key_name_2):
                X_1, y_1, plf_interval_1 = dict_funcs[key_name_1]
                X_2, y_2, plf_interval_2 = dict_funcs[key_name_2]

                # draw_d_and_d_interval(key_name_1, key_name_2, X_1, X_2,
                #                       reg_func_d_interval_1=plf_interval_1,
                #                       reg_func_d_interval_2=plf_interval_2,
                #                       degree=degree_now)
                # draw_two_p_and_d_interval(key_name_1, key_name_2, X_1, X_2,
                #                           y_1, y_2, plf_interval_1, plf_interval_2, count,
                #                           dp_num_1=10, dp_num_2=10, degree=degree_now)
                # count += 1
                if draw_two_p_and_d_interval_commX(key_name_1, key_name_2, X_1, X_2,
                                                y_1, y_2, plf_interval_1, plf_interval_2,
                                                degree_now, count) != 1:
                    count += 1
    print(count)