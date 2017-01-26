import numpy as np
from sklearn.linear_model import LinearRegression

Xi=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78,3.33,5.2,5.62,7.1,7.29,7.88,5.53])
Yi=np.array([7.01,2.78,5.47,6.71,4.1,4.23,4.05,3.88,5.5,5.22,6.23,7.2,6.57,5.2])
# Xi=np.array([1,2,3,4])
# Yi=np.array([6,5,7,10])
Xi = Xi.reshape(Xi.shape[0], 1)

reg_func = LinearRegression()
reg_func.fit(Xi, Yi)
X_curve = np.linspace(min(Xi), max(Xi))
X_curve = X_curve.reshape(X_curve.shape[0], 1)
y_curve = reg_func.predict(X_curve)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,9))
plt.title('Demo of Least Squares')
plt.xlabel('X')
plt.ylabel('y')
for item_X, item_Y in zip(Xi,Yi):
    plt.plot([item_X[0],item_X[0]],[item_Y,reg_func.predict(item_X)[0]], color='blue', linewidth=3)
plt.scatter(Xi,Yi,color="red",label="Sample Point",linewidth=3)
plt.plot(X_curve, y_curve, color="orange", label="Fitting Line", linewidth=2)
plt.legend()
plt.show()