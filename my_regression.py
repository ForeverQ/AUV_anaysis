from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import linear_model

class PolynomialRegressionInterval(object):
    """
    implement a way to do the interval regression

    Attributes
    ----------
    linear_reg_funcs_: list, store a series of tuples with functions and marks for different intervals
    and first element is a LinearRegression function of polynomial or linear, second element is a string of 'poly' or 'linear'


    linear_reg_derivatives_: list, store a list of tuples and
    first element is a derivative in the form of coef_, second element is a string of 'linear' or 'poly'

    linear_reg_derivatives_czpoints_: list, store a list of tuples.
    First element is the value of time and second value is a string of maximum or minimum

    intervals_: list, store a series of single float which represents the interval
    corresponding to the linear_reg_func.
    The number in it stands for the end of the subscript

    interval_values_quad_: list, store a series of numpy arrays corresponding to the intervals_
    the numpy arrays in it stands for the exact end values of the intervals

    """

    def __init__(self):
        self.linear_reg_funcs_ = []
        self.linear_reg_derivatives_ = []
        self.linear_reg_derivatives_czpoints_ = []
        self.intervals_ = []
        self.interval_values_quad_ = []

    def fit(self, X_quad, y, reg_type='linear', rsq_threshold_poly=0.99, mixed_mode=True, rsq_threshold_linear=0.99):
        """
        Fit and create different intervals.
        Create intervals by dichotomy with the threshold of r_square

        Parameters
        ----------
        X_quad : the whole X values after polynomial transform

        y : the whole y values

        rsq_threshold_poly : the threshold of r_squared to decide the
        interval dividing

        mixed_mode : means if the fitting processing mixed with linear fitting

        Returns
        -------
        self : returns an instance of self.
        """

        self.X_quad_when_fitting = X_quad

        # use a kind of small interval unit to force the fitting accurate
        interval_unit_num = len(self.X_quad_when_fitting) // 16

        # non-recursive version
        sub_start = 0
        sub_end = sub_start + interval_unit_num
        while sub_start < len(self.X_quad_when_fitting):
            r_square_poly = 0
            reg_func_poly = LinearRegression()
            has_been_in_while = False
            if mixed_mode:
                r_square_linear = 0
                reg_func_linear = LinearRegression()
                while r_square_linear < rsq_threshold_linear and r_square_poly < rsq_threshold_poly:
                    if has_been_in_while:
                        sub_end = (sub_end - sub_start) // 2 + sub_start
                    now_X_poly = self.X_quad_when_fitting[sub_start:sub_end]
                    now_X_linear = np.array([*map(lambda a:[a[1]], now_X_poly)])
                    now_y = y[sub_start:sub_end]
                    reg_func_poly = LinearRegression()
                    reg_func_linear = LinearRegression()
                    reg_func_poly.fit(now_X_poly, now_y)
                    reg_func_linear.fit(now_X_linear, now_y)
                    r_square_poly = reg_func_poly.score(now_X_poly, now_y)
                    r_square_linear = reg_func_linear.score(now_X_linear, now_y)
                    has_been_in_while = True
                    # TODO special situation when value y is stable r_square would be 0 after first calculation
                    if r_square_linear == 0:
                        r_square_linear = 1.0
                    elif r_square_poly == 0:
                        r_square_poly = 1.0
                self.intervals_.append(sub_end)
                self.interval_values_quad_.append(self.X_quad_when_fitting[sub_end - 1])
                # linear has an priority
                if r_square_linear >= rsq_threshold_linear:
                    self.linear_reg_funcs_.append((reg_func_linear,'linear'))
                    # for debug
                    print('linear_r2: ' + str(r_square_linear))
                else:
                    self.linear_reg_funcs_.append((reg_func_poly,'poly'))
                    # for debug
                    print('poly_r2: ' + str(r_square_poly))
                sub_start = sub_end
                sub_end = min(sub_start+interval_unit_num, len(self.X_quad_when_fitting))
            else:
                while r_square_poly < rsq_threshold_poly:
                    if has_been_in_while:
                        sub_end = (sub_end - sub_start) // 2 + sub_start
                    now_X_poly = self.X_quad_when_fitting[sub_start:sub_end]
                    now_y = y[sub_start:sub_end]
                    reg_func_poly = LinearRegression()
                    reg_func_poly.fit(now_X_poly, now_y)
                    r_square_poly = reg_func_poly.score(now_X_poly, now_y)
                    has_been_in_while = True
                    # TODO special situation when value y is stable r_square would be 0 after first calculation
                    if r_square_poly == 0:
                        r_square_poly = 1.0
                self.intervals_.append(sub_end)
                self.interval_values_quad_.append(self.X_quad_when_fitting[sub_end - 1])
                self.linear_reg_funcs_.append((reg_func_poly,'poly'))
                sub_start = sub_end
                sub_end = min(sub_start+interval_unit_num, len(self.X_quad_when_fitting))

        # # recursive version
        # def regression_interval(sub_start, sub_end):
        #     if sub_end <= sub_start:
        #         return 0
        #     else:
        #         now_X = self.X_quad_when_fitting[sub_start:sub_end]
        #         now_y = y[sub_start:sub_end]
        #         if reg_type=='ridge':
        #             # reg_func = linear_model.Ridge(alpha=0.0001)
        #             reg_func = linear_model.Ridge(alpha=1.0)
        #             reg_func.fit(X_quad, y)
        #             r_square = reg_func.score(now_X, now_y)
        #         else:
        #             reg_func = LinearRegression()
        #             reg_func.fit(now_X, now_y)
        #             r_square = reg_func.score(now_X, now_y)
        #         if r_square < rsq_threshold:
        #             sub_end = (sub_end - sub_start) // 2 + sub_start
        #             regression_interval(sub_start, sub_end)
        #         else:
        #             self.intervals_.append(sub_end)
        #             self.interval_values_quad_.append(self.X_quad_when_fitting[sub_end - 1])
        #             self.linear_reg_funcs_.append(reg_func)
        #             regression_interval(sub_start=sub_end, sub_end=len(self.X_quad_when_fitting))
        #
        # regression_interval(0, len(self.X_quad_when_fitting))

        # for debug
        print()
        return self

    def predict(self, X_quad):
        # TODO solve long time
        """

        Parameters
        ----------
        param X_quad: the X values used to calculate the function y

        Returns
        -------
        return: array, shape = (n_samples,)
                predicted values
        """

        # This former algorithm is too slow
        # y = []
        # reg_func_interval_num = 0
        # # get the value of time ** 1 from X_quad
        # interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        # for index, one_X in enumerate(X_quad):
        #     # get the value of time ** 1 from X_quad
        #     value_of_X = one_X[1]
        #     while not (reg_func_interval_num >= len(self.intervals_) - 1):
        #         if value_of_X < interval_right_value:
        #             is_in_range = True
        #             break
        #         reg_func_interval_num += 1
        #         interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        #     a_sub_y = self.linear_reg_funcs_[reg_func_interval_num].predict(X_quad[index:index+1])
        #     y += list(a_sub_y)

        y = []
        reg_func_interval_num = 0
        interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        X_quad_left_index = 0
        value_of_X = X_quad[0][1]
        while not (value_of_X <= interval_right_value):
            reg_func_interval_num += 1
            interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        for index, one_X in enumerate(X_quad):
            # get the value of time ** 1 from X_quad
            value_of_X = one_X[1]
            if (index == len(X_quad)-1):
                now_X_quad = X_quad[X_quad_left_index:index+1]
                now_func = self.linear_reg_funcs_[reg_func_interval_num][0]
                if self.linear_reg_funcs_[reg_func_interval_num][1]=='linear':
                    now_X_linear = np.array([*map(lambda a:[a[1]],now_X_quad)])
                    sub_y = now_func.predict(now_X_linear)
                else:
                    sub_y = now_func.predict(now_X_quad)
                y += list(sub_y)
            elif (value_of_X > interval_right_value):
                now_X_quad = X_quad[X_quad_left_index:index]
                if self.linear_reg_funcs_[reg_func_interval_num][1]=='linear':
                    now_X_linear = np.array([*map(lambda a:[a[1]],now_X_quad)])
                    sub_y = self.linear_reg_funcs_[reg_func_interval_num][0].predict(now_X_linear)
                else:
                    sub_y = self.linear_reg_funcs_[reg_func_interval_num][0].predict(now_X_quad)
                y += list(sub_y)
                X_quad_left_index = index
                reg_func_interval_num += 1
                interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        return np.array(y)

    def calculate_derivatives(self):
        def my_derivative_poly(coef_):
            coef = []
            for i in range(1, len(coef_)):
                coef.append(i*coef_[i])
            return np.array(coef)
        for func_tuple in self.linear_reg_funcs_:
            if func_tuple[1] == 'linear':
                coef_d = np.array([func_tuple[0].coef_[0]])
                self.linear_reg_derivatives_.append((coef_d, 'linear'))
            else:
                coef_d = my_derivative_poly(func_tuple[0].coef_)
                self.linear_reg_derivatives_.append((coef_d, 'poly'))

    def predict_d(self, X_quad_d):
        if len(self.linear_reg_derivatives_) == 0:
            self.calculate_derivatives()
        y_d = []
        reg_func_interval_num = 0
        # get the value of time ** 1 from X_quad
        interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        X_quad_left_index = 0
        is_in_range = False
        for index, one_X_d in enumerate(X_quad_d):
            # get the value of time ** 1 from X_quad
            value_of_X_d = one_X_d[1]
            if value_of_X_d < interval_right_value:
                is_in_range = True
            if is_in_range:
                if (value_of_X_d > interval_right_value):
                    if (index != len(X_quad_d)-1):
                        now_X_quad = X_quad_d[X_quad_left_index:index]
                        sub_coef_d = self.linear_reg_derivatives_[reg_func_interval_num][0]
                        if self.linear_reg_derivatives_[reg_func_interval_num][1] == 'linear':
                            sub_y_d = len(now_X_quad)*list(sub_coef_d)
                        else:
                            sub_y_d = now_X_quad.dot(sub_coef_d)
                        y_d += list(sub_y_d)
                        X_quad_left_index = index
                        reg_func_interval_num += 1
                        interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
                    else:
                        now_X_quad = X_quad_d[X_quad_left_index:index+1]
                        sub_coef_d = self.linear_reg_derivatives_[reg_func_interval_num][0]
                        if self.linear_reg_derivatives_[reg_func_interval_num][1] == 'linear':
                            sub_y_d = len(now_X_quad)*list(sub_coef_d)
                        else:
                            sub_y_d = now_X_quad.dot(sub_coef_d)
                        y_d += list(sub_y_d)
                elif (index == len(X_quad_d)-1):
                    now_X_quad = X_quad_d[X_quad_left_index:index+1]
                    sub_coef_d = self.linear_reg_derivatives_[reg_func_interval_num][0]
                    if self.linear_reg_derivatives_[reg_func_interval_num][1] == 'linear':
                        sub_y_d = len(now_X_quad)*list(sub_coef_d)
                    else:
                        sub_y_d = now_X_quad.dot(sub_coef_d)
                    y_d += list(sub_y_d)
            else:
                reg_func_interval_num += 1
                interval_right_value = self.X_quad_when_fitting[self.intervals_[reg_func_interval_num]-1][1]
        return np.array(y_d)

    def calculate_czpoints(self):
        self.calculate_derivatives()
