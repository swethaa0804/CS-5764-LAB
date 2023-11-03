import operations as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensor as tf
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
np.random.seed(5764)
# mnist_test.csv
# Create a dataframe with 4 columns and 10 observations usinf np.random.rand and call it as a,b,c,d

df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
print(df.head())

#plot the dataframe
explode = (0.01,0.01,0.05,0.01)
df.plot(subplots = True , layout = (4,1),title='dummy dataset',xlabel = 'x-axis', ylabel = 'y-axis', grid = True, fontsize = 12, xlim = (0,20),ylim = (0,3),
        figsize = (10,10)
        )
plt.tight_layout()
plt.show()

# non time series to time series using pd.date_range command
# plot(Line plot) the total bill and tip versus time
# start date Jan 1 2000, freq Daily

date = pd.date_range(start= '1-1-2000' , periods= len(tips) ,freq= 'D')
type(date)
tips.index = date
tips[['tip','total_bill']].plot( kind = 'line',title='tip dataset',xlabel = 'time', ylabel = 'USD($)', grid = True, fontsize = 12, figsize = (8,8))
plt.tight_layout()
plt.show()

# Bar plot is used for categorical features
# Next to each other
df.plot(kind = 'bar',
        )
plt.show()

#bar on top of each other
df.plot(kind = 'bar', stacked = True
        )
plt.show()

# Bar plot xaxis - thurs-sat height: sum of the total of the bill. which day paid more? weekend-sator sun ascending/descending

# Create a bar plot(vertical) x-axis should be unique time
# y-axis should represent the sum of total of the bill for each day
# ascending order

tips_group = tips.groupby(['day']).sum()
 # tips_group.plot(kind='bar') doesnt work as day are in index
tips_group = tips_group.reset_index()
tips_group = tips_group.sort_values(by = 'total_bill', ascending=True)

tips_group.plot(kind = 'bar' ,title='bar plot dataset', x = 'day', y = 'total_bill',xlabel = 'day', ylabel = 'USD($)', grid = True, fontsize = 12)

# who is generous

# Histogram plot - numerical , increase bin reduce the length

tips[['tip','total_bill']].plot(kind = 'hist' , bins = 50, grid = True, stacked = False, alpha = 0.5, orientation = 'horizontal', title = 'hist plot')
plt.tight_layout()
plt.show() # variance observation high freq,mean
# Transparency - overlap- aplha

# Box plot - anamoly, outliers

tips[['total_bill','tip', 'sex']].plot(kind = 'box', grid = True, title = 'Box plot',by = 'sex')
plt.tight_layout()
plt.show()

# Area plot - higlight an insignifcant band

df.plot(kind = 'area')
plt.tight_layout()
plt.show()


#Scatter
tips.plot(kind = 'scatter', x = 'total_bill', y = 'tip', color = 'red',marker = '*', s=100,grid = True,title = 'scatter plot')
plt.tight_layout()
plt.show()

# pie chart needs an aggregation
tips_group_time = tips.groupby(['day']).sum()
tips_group_time.plot(kind = 'pie', y = 'tip', autopct = '%1.0f%', explode = explode, startangle = 60)
plt.tight_layout()
plt.show()