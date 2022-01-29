# For data preprocess
import numpy as np
import csv
import os
from math import exp, log
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Fixed random seed
np.random.seed(0)

# Load dataset1 Gaussian
def dataset_gaussian(n):
    cov = [[1, 0], [0, 1]]
    data1 = np.random.multivariate_normal([-1, 1], cov, n)
    data2 = np.random.multivariate_normal([1, -1], cov, n)
    c1 = np.array([[1] * n])
    data1 = np.concatenate((c1.T, data1), axis=1) # Set data label as the first element in array
    c2 = np.array([[0] * n])
    data2 = np.concatenate((c2.T, data2), axis=1) # Set data label as the first element in array
    data = np.concatenate((data1, data2), axis=0)
    np.random.shuffle(data)
    return data

# Load dataset2 MNIST
def dataset_MNIST(path, chr):
    datadim = 10000 if path == 'mnist_test.csv' else 60000
    c1, c2 = chr
    with open(path, 'r') as fp:
        data = list(csv.reader(fp))
        idx = [i for i in range(1, datadim+1) if data[i][0] == c1 or data[i][0] == c2] # Choose the characters we want to classify, ex: '1' and '7'
        data = np.array(data)[idx, :].astype(float)
    for i in range(len(data)):
        data[i][0] = 1 if data[i][0] == int(c1) else 0 # Convert character c1 into 1 or c2 into 0
    return data

# sigmoid
def sigmoid(x):
    if x <= -709:
        return 0
    return 1/(1+exp(-x))

# 2-norm
def norm(x):
    return sum([i ** 2 for i in x]) ** 0.5

class MyClassifier:
    def __init__(self):
        self.theta = None # weight coefficient
        self.dim = None # 1 + the number of features
        self.x = None # The features of current chosen data
        self.y = None # The labels of current chosen data
        self.m = None # total # of samples
        self.n = None # current # of samples
        self.mean = None # current mean of all chosen samples
        self.std = None # current standard deviation of all chosen samples
    def sample_selection(self, training_sample): # Data label is in the first element of training_sample
        self.m, self.dim = np.shape(training_sample)
        nor_data = self.normalize(training_sample[:, 1:])
        def label_center():
            idx0, idx1 = [], []
            for i in range(self.m):
                if training_sample[i][0]:
                    idx1.append(i)
                else:
                    idx0.append(i)
            ret = []
            ret.append(np.mean(nor_data[idx0, :], axis=0))
            ret.append(np.mean(nor_data[idx1, :], axis=0))
            return ret
        p0, p1 = label_center() # find two center points of two labels
        v = p1 - p0
        dis = norm(v)
        c = -np.inner(v, (p0+p1)/2)
        idx = []
        for i in range(self.m):
            b = np.inner(v, nor_data[i, :])+c
            l = training_sample[i, 0]
            # Choose Method here
            # Uncomment here to use Method1 'Perpendicular-Bisector' and please comment Method2 'Distance-Ratio'
            # if ((b > 0 and l == 1) or (b <= 0 and l == 0)) and abs(b/(dis ** 2)) <= config['dis']:
            # Uncomment here to use Method2 'Distance-Ratio' and please comment Method1 'Perpendicular-Bisector'
            if ((b > 0 and l == 1) or (b <= 0 and l == 0)) and 1/config['divide'] <= norm(nor_data[i, :]-p0)/norm(nor_data[i, :]-p1) <= config['divide']:
                idx.append(i)
        self.theta = np.zeros(self.dim)
        self.n = len(idx)
        self.x = training_sample[idx, 1:]
        self.y = training_sample[idx, 0]
        c = np.array([[1]*self.n]) # constant
        training_data = self.normalize(self.x)
        training_data = np.concatenate((c.T, training_data), axis=1)
        self.train(training_data, self.y)
        return
    def train(self, train_data, train_label):
        for _ in range(config['iterate']): # parameter
            self.gd(train_data, train_label, self.n, self.theta)
        return
    def f(self, input):
        t = np.inner(self.theta, np.concatenate((np.array([1]).T,input),axis=0))
        return 1 if t > 0 else 0
    def test(self, input):
        input_size = len(input)
        ans = 0
        input[:, 1:] -= self.mean
        for i in range(1, len(input[0])):
            if self.std[i-1] != 0:
                input[:, i] /= self.std[i-1]
        for i in range(input_size):
            if self.f(input[i][1:]) == input[i][0]:
                ans += 1
        return ans/input_size
    def calculate_lost(self, x, y, n, theta):
        ans = 0
        for i in range(n):
            t = np.inner(theta, x[i])
            ans += (y[i] * log(sigmoid(t))) + (1-y[i]) * log(1-sigmoid(t))
        return -ans/n
    def gd(self, x, y, n, theta):
        t = np.array([sigmoid(np.inner(theta, x[i]))-y[i] for i in range(n)])
        theta -= config['lr'] * np.matmul(x.T, t)
    def normalize(self, vector):
        self.mean = np.mean(vector, axis=0)
        self.std = np.std(vector, axis=0)
        nor_data = vector - self.mean
        for i in range(len(nor_data[0])):
            if self.std[i] != 0:
                nor_data[:, i] /= self.std[i]
        return nor_data

# Change the parameters here
config = {
    'dis':0.25, # We decide to choose the data point whose ratio of the distance to the perpendicular and the distance of two center points is smaller than this 'dis' and in the right halfspace
    'divide':1.15, # We decide to choose the data point whose ratio of two distances to two center points is smaller than this 'divide' and larger than '1/divide'
    'iterate':100, # times of iteration
    'lr':0.001, # learning rate
    'classify':('1','7') # classify which two characters
}

# Load dataset
# Uncomment here to load dataset1 Gaussian and please comment dataset2 MNIST
data = dataset_gaussian(6000)
testdata = dataset_gaussian(1000)
# Uncomment here to load dataset2 MNIST and please comment dataset1 Gaussian
'''
data = dataset_MNIST('mnist_train.csv', config['classify'])
testdata = dataset_MNIST('mnist_test.csv', config['classify'])
'''

# Create Class MyClassifier
a = MyClassifier()
a.sample_selection(data) # Main function to run our program

# Print the results
print('Total Samples: {}'.format(a.m))
print('Total Chosen Samples: {}'.format(a.n))
print('Accuracy: {}'.format(a.test(testdata))) # Test

# Plot chosen sample points for Gaussian dataset
'''
idx1 = [i for i in range(a.n) if a.y[i] == 1]
idx2 = [i for i in range(a.n) if a.y[i] == 0]
plt.scatter(a.x[idx1, 0], a.x[idx1, 1])
plt.scatter(a.x[idx2, 0], a.x[idx2, 1])
plt.show()
'''

# ouput csv files
'''
pathb, pathw = 'bias_19_Part2_', 'weights_19_Part2_'
pathb += 'Gaussian.csv' if len(a.theta) == 3 else 'MNIST.csv'
pathw += 'Gaussian.csv' if len(a.theta) == 3 else 'MNIST.csv'
with open(pathb, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(['bias'])
    writer.writerow([a.theta[0]])
with open(pathw, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(['weights'])
    for p in a.theta[1:]:
        writer.writerow([p])
'''
