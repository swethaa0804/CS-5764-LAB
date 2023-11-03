import datetime as dt
import operations as pd
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import pingouin as pg
import pandas as pd

#Q1
from prettytable import PrettyTable
table = PrettyTable(["", "Sales", "AdBudget", "GDP"])
table.title = "Pearson Correlation Matrix for the tute1 dataset"
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url)
correlation_matrix = df[['Sales', 'AdBudget', 'GDP']].corr().round(2)
for i in correlation_matrix.index:
    correlation = correlation_matrix.loc[i].values.tolist()
    table.add_row([i] + correlation)
print(table)


#Q2
import operations as pd
from prettytable import PrettyTable
import pingouin as pg
data = df[["Sales", "AdBudget", "GDP"]]
table = PrettyTable(["", "Sales", "AdBudget", "GDP"])
table.title = "Partial Correlation Matrix for the tute1 dataset"
p1 = pg.partial_corr(data=df, x='Sales', y="AdBudget", covar="GDP")
p2 = pg.partial_corr(data=df, x='Sales', y="GDP", covar="AdBudget")
p3 = pg.partial_corr(data=df, x="AdBudget", y='Sales', covar="GDP")
p4 = pg.partial_corr(data=df, x="AdBudget", y="GDP", covar='Sales')
p5 = pg.partial_corr(data=df, x="GDP", y='Sales', covar="AdBudget")
p6 = pg.partial_corr(data=df, x="GDP", y="AdBudget", covar='Sales')
table.add_row(["Sales", "1", p1['r'].values[0].round(2), p2['r'].values[0].round(2)])
table.add_row(["AdBudget", p3['r'].values[0].round(2), "1", p4['r'].values[0].round(2)])
table.add_row(["GDP", p5['r'].values[0].round(2), p6['r'].values[0].round(2), "1"])
print(table)

# Q3

import numpy as np
import pandas as pd
from scipy.stats import t
import pingouin as pg
tute1 = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
corr_matrix = tute1.corr(method='pearson',numeric_only=True)
s_a = corr_matrix.loc['Sales', 'AdBudget']
s_g = corr_matrix.loc['Sales', 'GDP']
a_g = corr_matrix.loc['AdBudget', 'GDP']
partial_corr_s_a_g = pg.partial_corr(data=tute1, x='Sales', y='AdBudget', covar='GDP')
partial_corr_s_g_a = pg.partial_corr(data=tute1, x='Sales', y='GDP', covar='AdBudget')
partial_corr_a_s_g = pg.partial_corr(data=tute1, x='AdBudget', y='Sales', covar='GDP')
partial_corr_a_g_s = pg.partial_corr(data=tute1, x='AdBudget', y='GDP', covar='Sales')
partial_corr_g_s_a = pg.partial_corr(data=tute1, x='GDP', y='Sales', covar='AdBudget')
partial_corr_g_a_s = pg.partial_corr(data=tute1, x='GDP', y='AdBudget', covar='Sales')
n = tute1.shape[0]
t1 = s_a * np.sqrt(n - 2) / np.sqrt(1 - s_a**2)
t2 = s_g * np.sqrt(n - 2) / np.sqrt(1 - s_g**2)
t3 = a_g * np.sqrt(n - 2) / np.sqrt(1 - a_g**2)
df = n - 2
alpha = 0.05
critical_t = t.ppf(1 - alpha / 2, df)
print(f'T-test for Pearson correlation coefficient between Sales and AdBudget is: {t1:.2f}')
if np.abs(t1) > critical_t:
    print("The Pearson correlation between Sales and AdBudget is significant.")
else:
    print("The Pearson correlation between Sales and AdBudget is not significant.")

print(f'T-test for Pearson correlation coefficient between Sales and GDP is: {t2:.2f}')
if np.abs(t2) > critical_t:
    print("The Pearson correlation between Sales and GDP is significant.")
else:
    print("The Pearson correlation between Sales and GDP is not significant.")

print(f'T-test for Pearson correlation coefficient between AdBudget and GDP is: {t3:.2f}')
if np.abs(t3) > critical_t:
    print("The Pearson correlation between AdBudget and GDP is significant.")
else:
    print("The Pearson correlation between AdBudget and GDP is not significant.")

t4 = partial_corr_s_a_g['r'][0] * np.sqrt(n - 3) / np.sqrt(1 - partial_corr_s_a_g['r'][0]**2)
t5 = partial_corr_s_g_a['r'][0] * np.sqrt(n - 3) / np.sqrt(1 - partial_corr_s_g_a['r'][0]**2)
t6 = partial_corr_a_s_g['r'][0] * np.sqrt(n - 3) / np.sqrt(1 - partial_corr_a_s_g['r'][0]**2)
t7 = partial_corr_a_g_s['r'][0] * np.sqrt(n - 3) / np.sqrt(1 - partial_corr_a_g_s['r'][0]**2)
t8 = partial_corr_g_s_a['r'][0] * np.sqrt(n - 3) / np.sqrt(1 - partial_corr_g_s_a['r'][0]**2)
t9 = partial_corr_g_a_s['r'][0] * np.sqrt(n - 3) / np.sqrt(1 - partial_corr_g_a_s['r'][0]**2)

print(f'T-test for Partial correlation coefficient between Sales and AdBudget (controlling for GDP) is: {t4:.2f}')
if np.abs(t4) > critical_t:
    print("The Partial correlation between Sales and AdBudget (controlling for GDP) is significant.")
else:
    print("The Partial correlation between Sales and AdBudget (controlling for GDP) is not significant.")

print(f'T-test for Partial correlation coefficient between Sales and GDP (controlling for AdBudget) is: {t5:.2f}')
if np.abs(t5) > critical_t:
    print("The Partial correlation between Sales and GDP (controlling for AdBudget) is significant.")
else:
    print("The Partial correlation between Sales and GDP (controlling for AdBudget) is not significant.")

print(f'T-test for Partial correlation coefficient between AdBudget and Sales (controlling for GDP) is: {t6:.2f}')
if np.abs(t6) > critical_t:
    print("The Partial correlation between AdBudget and Sales (controlling for GDP) is significant.")
else:
    print("The Partial correlation between AdBudget and Sales (controlling for GDP) is not significant.")

print(f'T-test for Partial correlation coefficient between AdBudget and GDP (controlling for Sales) is: {t7:.2f}')
if np.abs(t7) > critical_t:
    print("The Partial correlation between AdBudget and GDP (controlling for Sales) is significant.")
else:
    print("The Partial correlation between AdBudget and GDP (controlling for Sales) is not significant.")

print(f'T-test for Partial correlation coefficient between GDP and Sales (controlling for AdBudget) is: {t8:.2f}')
if np.abs(t8) > critical_t:
    print("The Partial correlation between GDP and Sales (controlling for AdBudget) is significant.")
else:
    print("The Partial correlation between GDP and Sales (controlling for AdBudget) is not significant.")

print(f'T-test for Partial correlation coefficient between GDP and AdBudget (controlling for Sales) is: {t9:.2f}')
if np.abs(t9) > critical_t:
    print("The Partial correlation between GDP and AdBudget (controlling for Sales) is significant.")
else:
    print("The Partial correlation between GDP and AdBudget (controlling for Sales) is not significant.")


# Q5
from pandas_datareader import data
yf.pdr_override()

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.9, bottom=.5, right=.9, left=.1, hspace=.9, wspace=.3)
    df["High"].plot()
    plt.ylabel("High Price USD($)")
    plt.title("High price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#LOW
stocks = ['AAPL','ORCL','TSLA','IBM','YELP','MSFT']
start_date = dt.datetime(2000,1,1)
end_date = dt.datetime(2023,8,28)
num = 1
plt.figure(figsize=(16,8))
for i in stocks:
    df = data.get_data_yahoo(i,start_date,end_date)
    plt.subplot(3,2,num)
    plt.subplots_adjust(top=.9,bottom=.5,right=.9,left=.1,hspace=.9,wspace=.3)
    df["Low"].plot()
    plt.ylabel("Low Price USD($)")
    plt.title("Low price History of " +i)
    plt.grid()
    num = num +1
plt.show()

#OPEN
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.9, bottom=.5, right=.9, left=.1, hspace=.9, wspace=.3)
    df["Open"].plot()
    plt.ylabel("Open Price USD($)")
    plt.title("Open price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#CLOSE

stocks = ['AAPL',' ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.9,bottom=.5, right=.9, left=.1, hspace=.9, wspace=.3)
    df["Close"].plot()
    plt.ylabel("Close Price USD($)")
    plt.title("Close price History of " + i)
    plt.legend()
    plt.grid()
    num = num + 1
plt.show()

#VOLUME
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.9, bottom=.5, right=.9, left=.1, hspace=.9, wspace=.3)
    df["Volume"].plot()
    plt.ylabel("Volume Price USD($)")
    plt.title("Volume price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#ADJ CLOSE

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.9, bottom=.5, right=.9, left=.1, hspace=.9, wspace=.3)
    df["Adj Close"].plot()
    plt.ylabel("Adj Close Price USD($)")
    plt.title("Adj Close price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#Q7
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.95, hspace=.5)
    plt.hist(df["High"], bins=50)
    plt.xlabel("Value in USD($)")
    plt.ylabel("Frequency")
    plt.title("High price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#Q8
#LOW
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.95, hspace=.5)
    plt.hist(df["Low"], bins=50)
    plt.xlabel("Value in USD($)")
    plt.ylabel("Frequency")
    plt.title("Low price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#OPEN
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.95, hspace=.5)
    plt.hist(df["Open"],bins=50)
    plt.xlabel("Value in USD($)")
    plt.ylabel("Frequency")
    plt.title("Open price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#CLOSE
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.95, hspace=.5)
    plt.hist(df["Close"], bins=50)
    plt.xlabel("Value in USD($)")
    plt.ylabel("Frequency")
    plt.title("Close price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#VOLUME
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.95, hspace=.5)
    plt.hist(df["Volume"], bins=50)
    plt.xlabel("Value in USD($)")
    plt.ylabel("Frequency")
    plt.title("Volume price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

#ADJ CLOSE
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
num = 1
plt.figure(figsize=(16, 8))
for i in stocks:
    df = data.get_data_yahoo(i, start_date, end_date)
    plt.subplot(3, 2, num)
    plt.subplots_adjust(top=.95, hspace=.5)
    plt.hist(df["Adj Close"], bins=50)
    plt.xlabel("Value in USD($)")
    plt.ylabel("Frequency")
    plt.title("Adj Close price History of " + i)
    plt.grid()
    num = num + 1
plt.show()

# Q9
import pandas as pd
yf.pdr_override()
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
df = pd.DataFrame(data.get_data_yahoo('AAPL', start_date, end_date))
pd.plotting.scatter_matrix(df, diagonal='kde', alpha=0.5, s=10, hist_kwds={'bins': 50}, figsize=(16, 8))
plt.suptitle('Scatter Matrix for AAPL', fontsize=16)
plt.show()


# ORCL
yf.pdr_override()
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
df = pd.DataFrame(data.get_data_yahoo('ORCL', start_date,end_date))
pd.plotting.scatter_matrix(df,diagonal='kde', alpha=0.5, s=10, hist_kwds={'bins': 50}, figsize=(16, 8))
plt.suptitle('Scatter Matrix for ORCL', fontsize=16)
plt.show()

# TSLA
yf.pdr_override()
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
df = pd.DataFrame(data.get_data_yahoo('TSLA', start_date,end_date))
pd.plotting.scatter_matrix(df,diagonal='kde', alpha=0.5, s=10, hist_kwds={'bins': 50}, figsize=(16, 8))
plt.suptitle('Scatter Matrix for TSLA', fontsize=16)
plt.show()

#IBM
yf.pdr_override()
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
df = pd.DataFrame(data.get_data_yahoo('IBM', start_date, end_date))
pd.plotting.scatter_matrix(df, diagonal='kde', alpha=0.5, s=10, hist_kwds={'bins': 50}, figsize=(16, 8))
plt.suptitle('Scatter Matrix for IBM', fontsize=16)
plt.show()


#YELP
yf.pdr_override()
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
df = pd.DataFrame(data.get_data_yahoo('YELP', start_date, end_date))
pd.plotting.scatter_matrix(df,diagonal='kde', alpha=0.5, s=10, hist_kwds={'bins': 50}, figsize=(16, 8))
plt.suptitle('Scatter Matrix for YELP', fontsize=16)
plt.show()

#MSFT
yf.pdr_override()
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2023, 8, 28)
df = pd.DataFrame(data.get_data_yahoo('MSFT', start_date, end_date))
pd.plotting.scatter_matrix(df,diagonal='kde', alpha=0.5, s=10, hist_kwds={'bins': 50}, figsize=(16, 8))
plt.suptitle('Scatter Matrix for MSFT', fontsize=16)
plt.show()
