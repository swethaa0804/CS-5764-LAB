import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.dates as dates
# Question1
mean_x = float(input("Enter the mean 1: "))
variance_x = float(input("Enter the variance 1: "))
mean_y = float(input("Enter the mean 2: "))
variance_y = float(input("Enter the variance 2: "))
N = int(input("Enter the number of observations: "))
np.random.seed(5764)
x = np.random.normal(mean_x, np.sqrt(variance_x), N)
y = np.random.normal(mean_y, np.sqrt(variance_y), N)
print(f'random var. x = {x[0]:.2f}\n')
print(f'random var. y = {y[0]:.2f}')


# Question2
# mean_x = float(input("Enter the mean 1: "))
# variance_x = float(input("Enter the variance 1: "))
# mean_y = float(input("Enter the mean 2: "))
# variance_y = float(input("Enter the variance 2: "))
# N = int(input("Enter the number of observations: "))
# np.random.seed(5764)
# x = np.random.normal(mean_x, np.sqrt(variance_x), N)
# y = np.random.normal(mean_y, np.sqrt(variance_y), N)
num = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(np.sum((y - mean_y) ** 2))
pearsoncoefficient = num / denominator
print(f"Pearson's Correlation Coefficient: {pearsoncoefficient:.2f}")
# Question3
# mean_x = float(input("Enter the mean 1: "))
# variance_x = float(input("Enter the variance 1: "))
# mean_y = float(input("Enter the mean 2: "))
# variance_y = float(input("Enter the variance 2: "))
# N = int(input("Enter the number of observations: "))
# np.random.seed(5764)
# x = np.random.normal(mean_x, np.sqrt(variance_x), N)
# y = np.random.normal(mean_y, np.sqrt(variance_y), N)
# num = np.sum((x - mean_x) * (y - mean_y))
# denominator = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(np.sum((y - mean_y) ** 2))
pearson_coefficient = num / denominator
sample_mean_x = np.mean(x)
sample_mean_y = np.mean(y)
sample_variance_x = np.var(x)
sample_variance_y = np.var(y)
print(f"The sample mean of random variable x is: {sample_mean_x:.2f}")
print(f"The sample mean of random variable y is: {sample_mean_y:.2f}")
print(f"The sample variance of random variable x is: {sample_variance_x:.2f}")
print(f"The sample variance of random variable y is: {sample_variance_y:.2f}")
print(f"The sample Pearson’s correlation coefficient between x & y is: {pearson_coefficient:.2f}")
# Question4
# mean_x = float(input("Enter the mean 1: "))
# variance_x = float(input("Enter the variance 1: "))
# mean_y = float(input("Enter the mean 2: "))
# variance_y = float(input("Enter the variance 2: "))
# N = int(input("Enter the number of observations: "))
# np.random.seed(5764)
# x = np.random.normal(mean_x, np.sqrt(variance_x), N)
# y = np.random.normal(mean_y, np.sqrt(variance_y), N)
plt.plot(figsize=(10, 5))
plt.plot(x, label='x', color='orange')
plt.plot(y, label='y', color='blue')
plt.xlabel('Data Points')
plt.ylabel('Value')
plt.title('Line Plots of Random Variables x and y')
plt.legend()
plt.show()
# Question5
# mean_x = float(input("Enter the mean 1: "))
# variance_x = float(input("Enter the variance 1: "))
# mean_y = float(input("Enter the mean 2: "))
# variance_y = float(input("Enter the variance 2: "))
# N = int(input("Enter the number of observations: "))
# np.random.seed(5764)
# x = np.random.normal(mean_x, np.sqrt(variance_x), N)
# y = np.random.normal(mean_y, np.sqrt(variance_y), N)
plt.plot()
plt.hist(x, bins=60, label='x')
plt.hist(y, bins=60, label='y')
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.title('Histogram Plots of Random Variables x and y')
plt.grid()
plt.legend()
plt.show()
# Question6
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url)
print(df.head())
# Question7
u = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(u)
a = df['Sales']
b = df['AdBudget']
c = df['GDP']
corr1, _ = pearsonr(a, b)
corr2, _ = pearsonr(a, c)
corr3, _ = pearsonr(b, c)
print(f'Pearson correlation between Sales & AdBudget: %.2f' % corr1)
print(f'Pearson correlation between Sales & GDP: %.2f' % corr2)
print(f'Pearson correlation between AdBudget & GDP: %.2f' % corr3)
# Question8a
u = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
data = pd.read_csv(u)
a = data['Sales']
b = data['AdBudget']
corr, _ = pearsonr(a, b)
print(' The sample Pearson’s correlation coefficient between Sales & AdBudget  is: %.2f' % corr)
# Question8b
u = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
data = pd.read_csv(u)
a = data['Sales']
b = data['GDP']
corr, _ = pearsonr(a, b)
print('The sample Pearson’s correlation coefficient between Sales & GDP is: %.2f' % corr)
# Question8c
u = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
data = pd.read_csv(u)
a = data['AdBudget']
b = data['GDP']
corr, _ = pearsonr(a, b)
print(' The sample Pearson’s correlation coefficient between AdBudget & GDP is: %.2f' % corr)
# Question9
# url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])
x = df['Date']
y1 = df['Sales']
y2 = df['AdBudget']
y3 = df['GDP']
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Sales', marker='.')
plt.plot(x, y2, label='AdBudget', marker='s')
plt.plot(x, y3, label='GDP', marker='o')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sales, AdBudget, and GDP Over Time')
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=15))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# Question10
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url)
plt.figure()
y1 = df['Sales']
y2 = df['AdBudget']
y3 = df['GDP']
plt.hist(y1, bins=20, alpha=0.7, color='blue', label='Sales')
plt.hist(y2, bins=20, alpha=0.7, color='green', label='AdBudget')
plt.hist(y3, bins=20, alpha=0.7, color='red',  label='GDP')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histograms of Sales, AdBudget, and GDP')
plt.legend()
plt.tight_layout()
plt.show()


