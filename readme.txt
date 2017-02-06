Document of AUV Data Analysis
=============================

Environment
-----------

+ Python 3.5.1
+ pip 9.0.1
+ matplotlib (1.5.3)
+ numpy (1.11.0)
+ pandas (0.18.1)
+ scikit-learn (0.17.1)
+ scipy (0.17.1)
+ seaborn (0.7.1)


File Usages
------------

+ value_value_generated.py: draws combinitions of two generated functions

+ util.py: contains some functions can be treated as tools.

+ pre_process.py: does the job of pre-processing

+ linear_regression.py: conducts linear regression on data points

+ polynomial_regression.py: draws the results of pure polynomial regression on single kind of data points

+ my_regression.py: implements the interval polynomial regression

+ value_value_new.py: draws combinitions of two kinds of data values on one picture

+ polynomial_regression_interval.py: draws pictures of interval polynomial regression

+ polys_derivatives_PRI.py: draws comparisons of polys. and derivatives of two kinds of data values, which shares the same X axies. (based on polynomial regression interval)

+ points_polys_derivatives_PRI.py: draws comparisons of data points, polys. and derivatives together with cross-zero points of two kinds of data values. (based on polynomial regression interval)

+ two_derivatives_PRI.py: draws comparisons of two derivatives together with cross-zero points of two kinds of data values. (based on polynomial regression interval)


Steps
-----

1. go to file 'pre_process.py' to change the file path of one-dimention data and mult-dimention data
2. run the code by using command like 'python [filename]'. For example, 'python value_value_new.py'