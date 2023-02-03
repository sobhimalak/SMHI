import csv
import numpy as np
from matplotlib import pyplot as plt
import assests as functions

Next_Row = '\n'

# read csv from functions file
reader = functions.CSVReader('Smhi-data.csv')
reader.read()
data = reader.data

# crate array
Month = data[:, 0]
Actual_Temp = data[:, 1]
MedelTemp_Min = data[:,3]
MedelTemp_Max = data[:,2]

# crate matrix kx+m; k=1, m=1
mean_temp = functions.TempData(Actual_Temp,MedelTemp_Min,MedelTemp_Max)

#MM = functions.Temp
# print('function: \n', MM.max)
# print('function: \n', MM.min)
# print(MM.Mean)
print(mean_temp.max_temp)

Y = np.array(Month, dtype = float)
#print(Y)

At = mean_temp.actual_temp.T
A = np.dot(At, mean_temp.actual_temp)
Y = np.dot(At, Y)
#print(At)

# calling Gaussian_elimination from functions file
GE = functions.Gaussian_elimination()
x = GE.gaussElim(A, Y)
print('Actual_temp : ',Next_Row, x)

plt.plot(Month, Actual_Temp, 'r:', label= 'Temp')
plt.plot(Month, x[0] * Month + x[1], 'b', label= 'Linear Regression')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.title('Actual Temperature vs. Month')
plt.legend()
plt.show()

