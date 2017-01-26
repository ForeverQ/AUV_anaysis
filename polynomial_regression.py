import pre_treat as p_t
import linear_regression as l_r
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

def calculate_ratio(a_standard_tv):
    time_delt = a_standard_tv[-1][0] - a_standard_tv[0][0]
    delt_string_list = list(str(int(time_delt)))
    if delt_string_list[0]=='1':
        return len(delt_string_list) - 1
    else:
        return len(delt_string_list)

def my_derivative_poly(coef_):
    coef = []
    for i in range(1, len(coef_[0])):
        coef.append(i*coef_[0][i])
    return np.array(coef)

def generate_polynomials(X, degree):
    quad_feature = PolynomialFeatures(degree=degree)
    X_quad = quad_feature.fit_transform(X)
    return X_quad

def draw_points_and_poly(key_name, X, y, reg_func, degree, count):
    import matplotlib.pyplot as plt
    X_quad = generate_polynomials(X, degree)
    plt.figure(figsize=(16,9))
    plt.xlabel('time')
    plt.ylabel(key_name)
    plt.grid(True)
    plt.scatter(X, y, color='red', label='Sample Point', linewidths=1)
    # plt.plot(X, y)
    plt.plot(X, reg_func.predict(X_quad), color='orange', label='degree ' + str(degree), linewidth=3)
    plt.legend(loc='upper left')
    # plt.savefig('points_poly_data_1_ridge_' + str(count) + '.png', dpi=200)
    plt.show()
    plt.plot(np.linspace(0,0.5,100),[1]*100,color='black')

def print_poly_props(name, lin_reg, X_quad, y):
    print('Poly of ' + name + ':')
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
    print('r^2: ' + str(lin_reg.score(X_quad, y)))
    print()

def print_derivative_props(name, coef_d):
    print('Derivative of Poly ' + name + ':')
    print(coef_d)
    print()

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
    cross_zero_points = []
    point_delt_tuple_ppre = (*X[0], y[0], abs(y[0] - 0.0))
    point_delt_tuple_pre = (*X[1], y[1], abs(y[1] - 0.0))
    for (comm_xx, yy) in zip(X[2:], y[2:]):
        delt_now = abs(yy-0.0)
        if (point_delt_tuple_pre[2] < point_delt_tuple_ppre[2]) and \
                (point_delt_tuple_pre[2] < delt_now):
            if (point_delt_tuple_ppre[1]<0 and yy>0):
                cross_zero_points.append(point_delt_tuple_pre+('minimum',))
            elif (point_delt_tuple_ppre[1]>0 and yy<0):
                cross_zero_points.append(point_delt_tuple_pre+('maximum',))
        point_delt_tuple_ppre = point_delt_tuple_pre
        point_delt_tuple_pre = (*comm_xx, yy, delt_now)
    return cross_zero_points

def draw_two_p_and_d(key_name_1, key_name_2, X_1, X_2, lin_reg_1, lin_reg_2,
                     coef_d_1, coef_d_2, degree, count, t_comps_ratio):
    import matplotlib.pyplot as plt

    # find comm_x for picture
    comm_left = X_1[0] if X_1[0]>X_2[0] else X_2[0]
    comm_right = X_1[-1] if X_1[-1]<X_2[-1] else X_2[-1]
    comm_X = np.linspace(comm_left,comm_right,(comm_right-comm_left)*(10**t_comps_ratio))
    comm_X = comm_X.reshape(comm_X.shape[0], 1)
    comm_X_quad = generate_polynomials(comm_X, degree)

    # Derivative of this function for display
    comm_X_quad_d = generate_polynomials(comm_X, degree-1)

    fig, axarr = plt.subplots(2, sharex=True, figsize=(16,9))
    # fig.tight_layout()

    # plot 1
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel(key_name_1)
    axarr[0].grid(True)
    y_dp_1 = lin_reg_1.predict(comm_X_quad)
    y_1_range = np.amax(y_dp_1) - np.amin(y_dp_1)
    picture_y_1_max = np.amax(y_dp_1)+0.08*y_1_range
    picture_y_1_min = np.amin(y_dp_1)-0.08*y_1_range
    axarr[0].axis([comm_left,comm_right,picture_y_1_min,picture_y_1_max])
    axarr[0].plot(comm_X, y_dp_1, color='orange', label='degree '+str(degree), linewidth=3)

    # find extremum
    extremum_points = calculate_extremum(comm_X, y_dp_1)
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
    axarr[0].legend(loc='upper right')

    # plot 2
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel(key_name_2)
    axarr[1].grid(True)
    y_d_dp_2 = comm_X_quad_d.dot(coef_d_2)
    y_d_2_range = np.amax(y_d_dp_2) - np.amin(y_d_dp_2)
    picture_y_d_2_max = np.amax(y_d_dp_2)+0.08*y_d_2_range
    picture_y_d_2_min = np.amin(y_d_dp_2)-0.08*y_d_2_range
    axarr[1].axis([comm_left,comm_right,picture_y_d_2_min,picture_y_d_2_max])
    axarr[1].plot(comm_X, y_d_dp_2, color='c', label='derivative', linewidth=3)
    axarr[1].plot(comm_X, [0]*len(comm_X), color='black', label='zero', linewidth=1)

    # find cross 0 data points
    cross_zero_points = calculate_cross_zero(comm_X, y_d_dp_2)
    for index, item in enumerate(cross_zero_points):
        if item[3]=='maximum':
            axarr[1].scatter([item[0],],[item[1],], 30, color='blue')
            axarr[1].annotate('%.3f' % item[0],color='blue',
                              xy=(item[0], item[1]), xycoords='data',
                              xytext=(+10, +10), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='blue'))
            if index in (0,1):
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color='blue', label='cz_maximum', linewidth=1.5, linestyle="--")
            else:
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color='blue', linewidth=1.5, linestyle="--")
        elif item[3]=='minimum':
            axarr[1].scatter([item[0],],[item[1],], 30, color='g')
            axarr[1].annotate('%.3f' % item[0],color='g',
                              xy=(item[0], item[1]), xycoords='data',
                              xytext=(+10, +10), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='g'))
            if index in (0,1):
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color='g', label='cz_minimum', linewidth=1.5, linestyle="--")
            else:
                axarr[1].plot([item[0],item[0]],[item[1],picture_y_d_2_max], color='g', linewidth=1.5, linestyle="--")
    axarr[1].legend(loc='lower right')

    # plt.savefig('distance depth comparison' + str(count) + '.png', dpi=150)
    plt.show()


def make_polynomial_fitting_tv(tv_list, key_name, t_comps_ratio, degree=10,type='linear'):

    # for testing
    # key_name = ' PayloadPowerSwitch - Channel 7 (Ethernet Switch).Voltage.csv'
    # tv_list = dict_total[key_name]

    pd.set_option('precision',8)
    df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
    df_tv.sort_values(by='time')
    X_col_list = list(df_tv['time'])
    y_col_list = list(df_tv[key_name])

    # let the time value be smaller
    X_delta_sp = X_col_list[0] / (10**t_comps_ratio)
    X_col_list_cms = [*map(lambda a: a/(10**t_comps_ratio)-X_delta_sp, X_col_list)]

    # x_x = np.linspace(list(X_col)[0], list(X_col)[-1],len(X_col))
    X = np.array(X_col_list_cms).reshape(len(X_col_list_cms),1)
    y = np.array(y_col_list).reshape(len(y_col_list),1)
    X_quad = generate_polynomials(X, degree)
    if type=='linear':
        lin_reg = LinearRegression()
        lin_reg.fit(X_quad, y)
        return (X, X_quad, y, lin_reg)
    elif type=='ridge':
        clf = linear_model.Ridge(alpha=0.0001)
        clf.fit(X_quad, y)
        return (X, X_quad, y, clf)



def analyze_points_and_poly(dict_total, key_names, degree, t_comps_ratio):
    count = 0
    for key_name in key_names:
        if key_name.endswith('Distance.csv') or key_name.endswith('Depth.csv'):
            # Polynomial fitting
            X, X_quad, y, lin_reg = make_polynomial_fitting_tv(dict_total[key_name], key_name, t_comps_ratio, degree=degree)
            print_poly_props(key_name, lin_reg, X_quad, y)
            draw_points_and_poly(key_name, X, y, lin_reg, degree, count)
            count += 1


if __name__ == '__main__':
    dict_total = p_t.make_total_dict()

    # key_names in order by different types
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_notStable = [key_name for key_name in key_names_ordered if key_name not in l_r.STABLE_VALUES]

    degree_now = 100
    t_comps_ratio = calculate_ratio(a_standard_tv=dict_total[' x (m/s/s).Acceleration.csv'])

    analyze_points_and_poly(dict_total, key_names_ordered_notStable, degree_now, t_comps_ratio)
    exit(0)

    count = 0
    # analysis of two data polys and derivatives
    for key_name_1 in key_names_ordered_notStable:
        for key_name_2 in key_names_ordered_notStable:
            if (key_name_1.endswith('Distance.csv') and key_name_2.endswith('Depth.csv')) or \
                    key_name_2.endswith('Distance.csv') and key_name_1.endswith('Depth.csv'):

                # Polynomial fitting
                X_1, X_quad_1, y_1, lin_reg_1 = make_polynomial_fitting_tv(dict_total[key_name_1], key_name_1, t_comps_ratio, degree=degree_now)
                X_2, X_quad_2, y_2, lin_reg_2 = make_polynomial_fitting_tv(dict_total[key_name_2], key_name_2, t_comps_ratio, degree=degree_now)

                # A and b
                coef_1 = lin_reg_1.coef_
                coef_2 = lin_reg_2.coef_

                # Derivative of this function
                coef_d_1 = my_derivative_poly(coef_1)
                X_quad_d_1 = generate_polynomials(X_1, degree_now-1)
                coef_d_2 = my_derivative_poly(coef_2)
                X_quad_d_2 = generate_polynomials(X_2, degree_now-1)

                print_poly_props(key_name_1, lin_reg_1, X_quad_1, y_1)
                print_derivative_props(key_name_1, coef_d_1)
                print_poly_props(key_name_2, lin_reg_2, X_quad_2, y_2)
                print_derivative_props(key_name_2, coef_d_2)

                draw_two_p_and_d(key_name_1, key_name_2, X_1, X_2, lin_reg_1, lin_reg_2,
                                 coef_d_1, coef_d_2, degree_now, count, t_comps_ratio)
                count += 1



#
# About seaborn(not used)
# import seaborn as sns
# test_data = pd.DataFrame([[1, 1], [2, 4], [3, 9], [10, 100], [6, 36], [23, 529]], columns=['x', 'y'])
# sns.lmplot(x='x', y='y', data=test_data, order=2, ci=None, scatter_kws={"s": 80}, size=8, aspect=1.5)
# sns.lmplot(x='time', y=key_name, data=df_tv, order=2, ci=None, scatter_kws={"s": 30}, size=8, aspect=1.5)
# sns.plt.show()
# exit(0)