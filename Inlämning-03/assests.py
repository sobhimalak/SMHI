import numpy as np
import csv

class CSVReader:
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def read(self):
        with open(self.filename) as file:
            reader = csv.reader(file)
            next(reader)  # skip header row
            for row in reader:
                self.data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        self.data = np.array(self.data)

class Gaussian_elimination:
    def gaussElim(self, A, Y):
        n = len(Y)
        # Elimination phase
        for k in range(0, n - 1):
            for i in range(k + 1, n):
                if A[i, k] != 0.0:
                    # if not null define Î»
                    lam = A[i, k] / A[k, k]
                    # we calculate the new row of the matrix
                    A[i, k + 1:n] = A[i, k + 1:n] - lam * A[k, k + 1:n]
                    # we update vector b
                    Y[i] = Y[i] - lam * Y[k]
        # backward substitution
        for k in range(n - 1, -1, -1):
            Y[k] = (Y[k] - np.dot(A[k, k + 1:n], Y[k + 1:n])) / A[k, k]
        return Y

class TempData:
    def __init__(self, Actual_Temp,MedelTemp_Min,MedelTemp_Max):
        self.actual_temp = np.vstack([Actual_Temp, np.ones(len(Actual_Temp))]).T
        self.mid_temp = np.vstack([MedelTemp_Min, np.ones(len(MedelTemp_Min))]).T
        self.max_temp = np.vstack([MedelTemp_Max, np.ones(len(MedelTemp_Max))]).T
