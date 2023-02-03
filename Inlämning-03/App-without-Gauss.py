import csv
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = []
with open('Smhi-data.csv') as file:
    reader = csv.reader(file)
    next(reader)  # skip header row
    for row in reader:
        data.append([float(row[0]), float(row[1]), float(row[2]),float(row[3])])
data = np.array(data)


# Perform least squares fit
Month = data[:, 0]
Actual_Temp = data[:, 1]
MedelTemp_Max = data[:,2]
MedelTemp_Min = data[:,3]
A = np.vstack([Month, np.ones(len(Month))]).T
m, c = np.linalg.lstsq(A, Actual_Temp, rcond=None)[0]
print(m,c)


coefficients = np.polyfit(Month, Actual_Temp, 2)
a, b, c = coefficients
Month_prediction = np.array(range(13,25))
Temperature_prediction = a * Month_prediction**2 + b * Month_prediction + c

plt.plot(Month, Actual_Temp, 'g:.', label= 'Temperature')
plt.plot(Month_prediction, Temperature_prediction, 'b', label= 'Predicted Temperature')
plt.plot(Month, a * Month**2 + b * Month + c, 'r', label= 'Fitted polynomial')
plt.title('SMHI TEMP PREDICTION')
plt.xlabel("Month")
plt.ylabel("Temperature (C)")
plt.grid(True)

plt.legend()
plt.show()



#Plot data and fit line
# plt.plot(Month, Actual_Temp, 'g:.', label= 'Temperature')
# plt.plot(Month, m * Month + c, 'r', label= 'Fitted line')
# plt.title('SMHI TEMP PREDICTION')
# plt.xlabel("Month")
# plt.ylabel("Temperature (C)")
# plt.grid(True)
# plt.legend()
# plt.show()
