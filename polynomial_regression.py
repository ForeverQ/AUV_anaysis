import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np
import pre_process as p_p
import linear_regression as l_r
import util as t_l


VALUE_PRECISION = 8
DEGREE_NOW = 100


def draw_points_and_poly(key_name, X, y, reg_func, degree, count):
    """Draws both the data points and polynomial regression"""
    import matplotlib.pyplot as plt
    X_quad = t_l.generate_polynomials(X, degree)
    plt.figure(figsize=(16,9))
    plt.xlabel('time')
    plt.ylabel(key_name)
    plt.grid(True)
    plt.scatter(X, y, color='red', label='Sample Point', linewidths=1)
    plt.plot(X, reg_func.predict(X_quad), color='orange', label='degree ' + str(degree), linewidth=3)
    plt.legend(loc='upper left')
    # plt.savefig('points_poly_data_1_ridge_' + str(count) + '.png', dpi=200)
    plt.show()

def print_poly_props(name, lin_reg, X_quad, y):
    """Prints the coefficients of polynomial regression"""
    print('Poly of ' + name + ':')
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
    print('R^2: ' + str(lin_reg.score(X_quad, y)))
    print()

def make_polynomial_fitting_tv(tv_list, key_name, t_comps_ratio, degree=DEGREE_NOW,type='linear'):
    """Calculates the functions of polynomial regression

    Args:
        tv_list: lists of data values
        key_name: the name of data value for polynomial regression
        t_comps_ratio: the ratio of compression for timestamp
        degree: the degree of polynomial regression
        type: regression type ('linear' or 'ridge')

    Returns:
        A tuple of three elements:
            First element is a consequence of X
            Second element is polynomials generated according to the degree
            Third element is a consequence of y
            Fourth element is the function of regression
    """
    pd.set_option('precision', VALUE_PRECISION)
    df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
    df_tv.sort_values(by='time')
    X_col_list = list(df_tv['time'])
    y_col_list = list(df_tv[key_name])

    # Compress the value of timestamp
    X_delta_sp = X_col_list[0] / (10**t_comps_ratio)
    X_col_list_cms = [*map(lambda a: a/(10**t_comps_ratio)-X_delta_sp, X_col_list)]

    X = np.array(X_col_list_cms).reshape(len(X_col_list_cms),1)
    y = np.array(y_col_list).reshape(len(y_col_list),1)
    X_quad = t_l.generate_polynomials(X, degree)
    if type=='linear':
        lin_reg = LinearRegression()
        lin_reg.fit(X_quad, y)
        return (X, X_quad, y, lin_reg)
    elif type=='ridge':
        clf = linear_model.Ridge(alpha=0.0001)
        clf.fit(X_quad, y)
        return (X, X_quad, y, clf)


def analyze_points_and_poly(dict_total, key_names, degree, t_comps_ratio):
    """Conducts polynomial regression on each kind of data"""
    count = 0
    for key_name in key_names:
        # if key_name.endswith('Distance.csv') or key_name.endswith('Depth.csv'):
        X, X_quad, y, lin_reg = make_polynomial_fitting_tv(dict_total[key_name], key_name,
                                                           t_comps_ratio, degree=degree)
        print_poly_props(key_name, lin_reg, X_quad, y)
        draw_points_and_poly(key_name, X, y, lin_reg, degree, count)
        count += 1


if __name__ == '__main__':
    dict_total = p_p.make_total_dict()
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_notStable = [key_name for key_name in key_names_ordered if key_name not in l_r.STABLE_VALUES]
    degree_now = DEGREE_NOW
    t_comps_ratio = t_l.calculate_ratio(a_standard_tv=dict_total[' x (m/s/s).Acceleration.csv'])

    analyze_points_and_poly(dict_total, key_names_ordered_notStable, degree_now, t_comps_ratio)