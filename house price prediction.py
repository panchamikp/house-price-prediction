
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]).reshape(-1, 1)
y = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

model = LinearRegression()

model.fit(X, y)

predicted_price = model.predict([[2000]])
                                                                                                                                 
print("Predicted price for a 2000 sq. ft. house:", predicted_price[0])




