import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# loading the data from the cleaned up csv
csv = pd.read_csv('Cogs.csv')

hap2015 = csv['Happiness Score 2015'].to_numpy()
hap2016 = csv['Happiness Score 2016'].to_numpy()
hap2017 = csv['Happiness Score 2017'].to_numpy()
hap2018 = csv['Happiness Score 2018'].to_numpy()

hdi2015 = csv['HDI 2015'].to_numpy()
hdi2016 = csv['HDI 2016'].to_numpy()
hdi2017 = csv['HDI 2017'].to_numpy()
hdi2018 = csv['HDI 2018'].to_numpy()

# organizing data in a list of 2 respective data points
combinedHap = np.concatenate((hap2015, hap2016))
combinedHap = np.concatenate((combinedHap, hap2017))
combinedHap = np.concatenate((combinedHap, hap2018))

combinedHDI = np.concatenate((hdi2015, hdi2016))
combinedHDI = np.concatenate((combinedHDI, hdi2017))
combinedHDI = np.concatenate((combinedHDI, hdi2018))

X = np.array(combinedHDI).reshape((-1, 1))
Y = np.array(combinedHap).reshape((-1, 1))

# regression stuff
model = LinearRegression()
model.fit(X, Y)

r_sqrd = model.score(X, Y)
intercept = model.intercept_
slope = model.coef_

print("r^2 = " + str(r_sqrd))
print("Predicted Happiness = " + str(slope[0][0]) + "(HDI) + " + str(intercept[0]))

# scatter plot
m, b = np. polyfit(combinedHDI, combinedHap, 1)
plt.plot(combinedHDI, combinedHap, 'o')
plt.plot(combinedHDI, m * combinedHDI + b)

plt.xlabel('Human Development Index')
plt.ylabel('World Happiness Score')
plt.title('Human Development Index vs World Happiness Score')

plt.show()
