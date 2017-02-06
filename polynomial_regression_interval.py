import pandas as pd
import numpy as np
from scipy.ndimage import filters
from my_regression import PolynomialRegressionInterval
import pre_process as p_p
import linear_regression as l_r
import util as t_l

VALUE_PRECISION = 8
GAUSS_FILTER_PRECISION = 8
DEGREE_NOW = 6

def make_plf_cols(tv_list, key_name, t_comps_ratio, degree, GaussFiltered=True, GF_s=GAUSS_FILTER_PRECISION):
    """Conducts timestamp compression and GaussFilter on values

    Args:
        tv_list: data values
        key_name: name of this data
        t_comps_ratio: the compression ratio on timestamp
        degree: degree of polynomial
        GaussFiltered: if it is GaussFiltered
        GF_s: the precision of GaussFilter

    Returns:
        A tuple of three elements:
            First element is a consequence of X
            Second element is polynomials generated according to the degree
            Third element is a consequence of y
    """
    pd.set_option('precision', VALUE_PRECISION)
    df_tv = pd.DataFrame(tv_list, columns=['time', key_name])
    df_tv.sort_values(by='time')
    X_col_list = list(df_tv['time'])
    y_col_list = list(df_tv[key_name])
    X_delta_sp = X_col_list[0] // (10**t_comps_ratio)
    X_col_list_cms = [*map(lambda a: a/(10**t_comps_ratio)-X_delta_sp, X_col_list)]
    X = np.array(X_col_list_cms).reshape(len(X_col_list_cms),1)
    y = np.array(y_col_list).reshape(len(y_col_list),1)
    X_quad = t_l.generate_polynomials(X, degree)
    if GaussFiltered:
        y_list = [item[0] for item in y]
        y_GF = filters.gaussian_filter1d(y_list,GF_s)
        return (X, X_quad, np.array(y_GF))
    else:
        return (X, X_quad, y)


def draw_points_and_poly_interval(key_name, X, y, reg_func_interval, degree, count):
    """Draws both the points and interval poly. regression on one picture"""
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

    for x_interval in interval_values:
        plt.plot([x_interval,x_interval],[picture_y_min,picture_y_max], color='k', linewidth=1.5, linestyle="--")
    plt.legend(loc='upper left')
    # plt.savefig('points_poly_' + str(count) + '.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    dict_total = p_p.make_total_dict()
    key_names_ordered = sorted(dict_total, key=lambda item: item[::-1])
    key_names_ordered_notStable = [key_name for key_name in key_names_ordered if key_name not in l_r.STABLE_VALUES]
    t_comps_ratio = t_l.calculate_ratio(a_standard_tv=dict_total[' x (m/s/s).Acceleration.csv'])
    dict_funcs = dict()
    for key_name in key_names_ordered_notStable:
        X, X_quad, y = make_plf_cols(dict_total[key_name], key_name, t_comps_ratio,
                                               degree=DEGREE_NOW, GaussFiltered=True)
        plf_interval = PolynomialRegressionInterval()
        plf_interval.fit(X_quad, y)
        plf_interval.calculate_derivatives()
        dict_funcs[key_name] = (X, y, plf_interval)

    # Conducts interval regression
    count = 0
    for key_name in key_names_ordered_notStable:
        # if key_name.endswith('Distance.csv') or key_name.endswith('Depth.csv'):
        X, y, plf_interval = dict_funcs[key_name]
        draw_points_and_poly_interval(key_name, X, y, plf_interval, DEGREE_NOW, count)
        count += 1
