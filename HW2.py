import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv",header=[0,1])
df1 = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv",header=[1])
date = pd.date_range(start = '23-1-2020',
 periods = len(df),
 freq='D')
df.index = date
#Part 2 Csv fileS
data =pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/titanic.csv")
data2 =data.dropna()

countries =["United Kingdom","China","US","Germany","Brazil","India","Italy"]
mean =[]
variance=[]
median = []

#Question 1 code
df2 = pd.DataFrame(df).dropna()
df3 = df2.filter(like="China.")
print(df3)
print("Sum of China")
df2['China_sum']= df3.sum(axis=1).round(2)
print(df2)

#Question 2 code
df3 = df2.filter(like="United Kingdom")
print(df3)
print("Sum of United Kingdom")
print(df3.sum(axis=1).round(2))

#Question 3 code
df.index = date
df[['US']].plot(kind = 'line',
title = 'US Confirmed Covid 19 cases',
xlabel = 'Year',
ylabel = 'Confirmed Covid 19 cases',
grid = True,
fontsize = 10
 )
plt.legend(["US"])
plt.tight_layout()
plt.show()

#Question 4 code
df.index = date
df[["United Kingdom",'China','US',"Italy","Brazil","Germany","India"]].plot(kind = 'line',
title = 'Global Confirmed Covid 19 cases',
xlabel = 'Year',
ylabel = 'Confirmed Covid 19 cases',color =['C0','C1','C2','C3','C4','C5','C6'],
grid = True,
fontsize = 10
 )
plt.legend(['United Kingdom_Sum','China_Sum','US',"Italy","Brazil","Germany","India"])
plt.tight_layout()
plt.show()

#Question 5 code
us_data = df['US']['nan']
us_data = us_data.fillna(0)
plt.figure(figsize=(12, 6))
plt.hist(date,weights=us_data,bins=50)
plt.xlabel("Date")
plt.ylabel("Confirmed Covid 19 Cases")
plt.title("US Confirmed Covid 19 Cases")
plt.grid()
plt.tight_layout()
plt.show()

#Question 6 code
china = df["China"].sum(axis=1)
uk = df["United Kingdom"].sum(axis=1)
ger =df["Germany"]
brazil = df["Brazil"]
india = df["India"]
italy= df["Italy"]
fig, axs = plt.subplots(3, 2, figsize=(16, 8))

#UK_Sum
axs[0, 0].hist(date,weights = uk, bins=50)
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel("Confirmed Covid 19 Cases")
axs[0, 0].set_title("United Kingdom_Sum Confirmed Covid 19 Cases")
axs[0, 0].grid()
#China_Sum
axs[0, 1].hist(date,weights = china, bins=50)
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel("Confirmed Covid 19 Cases")
axs[0, 1].set_title("China_Sum Confirmed Covid 19 Cases")
axs[0, 1].grid()
#Italy
axs[1, 0].hist(date,weights = italy, bins=50)
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel("Confirmed Covid 19 Cases")
axs[1, 0].set_title("Italy Confirmed Covid 19 Cases")
axs[1, 0].grid()
#Brazil
axs[1, 1].hist(date,weights = brazil, bins=50)
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel("Confirmed Covid 19 Cases")
axs[1, 1].set_title("Brazil Confirmed Covid 19 Cases")
axs[1, 1].grid()
#Germany
axs[2, 0].hist(date,weights = ger, bins=50)
axs[2, 0].set_xlabel('Date')
axs[2, 0].set_ylabel("Confirmed Covid 19 Cases")
axs[2, 0].set_title("Germany Confirmed Covid 19 Cases")
axs[2, 0].grid()
#India
axs[2, 1].hist(date,weights = india, bins=50)
axs[2, 1].set_xlabel('Date')
axs[2, 1].set_ylabel("Confirmed Covid 19 Cases")
axs[2, 1].set_title("India Confirmed Covid 19 Cases")
axs[2, 1].grid()
plt.tight_layout()
plt.show()

#Question 7 code
for i in countries:
    mean.append(round(df[i].mean().max(), 2))
    variance.append(round(df[i].var().max(), 2))
    median.append(round(df[i].median().max(), 2))
print( "Highest Mean Value Country ",countries[mean.index(max(mean))])
print("Highest Variance Value Country ",countries[variance.index(max(variance))])
print("Highest Median Value Country ",countries[median.index(max(median))])
print( "Highest Mean Value Country ",countries[mean.index(max(mean))])
print("Highest Variance Value Country ",countries[variance.index(max(variance))])
print("Highest Median Value Country ",countries[median.index(max(median))])

#Part-2 Question 1 code
percentage_before = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100

cleaned_titanic_data = data.dropna()
cleaned_titanic_data.reset_index(inplace=True)
percentage_after = (cleaned_titanic_data.isnull().sum().sum() / (cleaned_titanic_data.shape[0] * cleaned_titanic_data.shape[1])) * 100

print("First 5 rows of the cleaned Titanic dataset:")
print(cleaned_titanic_data.head())

percentage_imputed = percentage_before - percentage_after
print(f"Percentage of data imputed: {percentage_imputed:.2f}%")

#Part-2 Question 2 code
labels ='Male','female'
data3 = data2["Sex"].value_counts()
total = data2["Sex"].value_counts().sum()
plt.pie(data3,labels=labels,explode=[.01,0],autopct=lambda p: '{:.0f}'.format(p * total / 100))
plt.title("pie chart of the people in titanic")
plt.legend()
plt.show()

#Part-2 Question 3 code
labels ='Male','female'
data3 = data2["Sex"].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title("pie chart of total  people in titanic in %")
plt.legend()
plt.show()

#Part-2 Question 4 code
labels ='Male Not survied','Male survied'
data2 =data.dropna()
data3 = data2.groupby('Sex').Survived.value_counts().male
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.2f%%')
plt.title("pie chart of Male survival in titanic")
plt.legend()
plt.show()

#Part-2 Question 5 code
labels ='Female survied','female Not survied'
data3 = data2.groupby('Sex').Survived.value_counts().female
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.2f%%')
plt.title("pie chart of Female survival in titanic")
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.show()

#Part-2 Question 6 code
labels ='Ticket class1','Ticket class2','Ticket class3'
data3 = data2["Pclass"].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.15,.15],autopct='%1.1f%%')
plt.title("pie chart passenger based on the level")
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.show()

# Part-2 Question 7 code
labels ='class1','class2','class3'
data3 = data2["Pclass"][data2["Survived"] ==1].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.15,.15],autopct='%1.1f%%')
plt.title("  Survival Rate based on the Ticket Class")
plt.legend()
plt.show()

#Part-2 Question 8 code
labels ='Survival rate','Death rate',
data3 = data2["Survived"][data2['Pclass'] ==1].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title(" survival & Death rate: Ticket class 1")
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
plt.show()


#Part-2 Question 9 code
labels ='Survival rate','Death rate'
data3 = data2["Survived"][data2['Pclass'] ==2].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title(" survival & Death rate: Ticket class 2")
plt.legend()
plt.show()

#Part-2 Question 10 code
labels ='Survival rate','Death rate'
data3 = data2["Survived"][data2['Pclass'] ==3].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title(" survival & Death rate: Ticket class 3")
plt.legend()
plt.show()

#Part-2 Question 11 code
plt.figure(figsize=(16,8))
plt.subplots_adjust(top=.9,hspace=.7,)
plt.subplot(331)
labels ='Male','female'
data3 = data2["Sex"].value_counts()
total = data2["Sex"].value_counts().sum()
plt.pie(data3,labels=labels,explode=[.01,0],autopct=lambda p: '{:.0f}'.format(p * total / 100))
plt.title("Pie chart of total people in titanic")
plt.legend(loc='upper right', bbox_to_anchor=(1.7, 0.9))

plt.subplot(332)
labels ='Male','female'
data3 = data2["Sex"].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title("Pie chart of total people in titanic in %")
plt.legend(loc='upper right', bbox_to_anchor=(1.7, 0.9),prop={'size':10})

plt.subplot(333)
labels ='Male Not survied','Male survied'
data3 = data2.groupby('Sex').Survived.value_counts().male
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title("Pie chart of Male survival in titanic")
plt.legend(loc='upper right', bbox_to_anchor=(2.2, 0.9),prop={'size':10})

plt.subplot(334)
labels ='Female  survied','female Not survied'
data3 = data2.groupby('Sex').Survived.value_counts().female
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title("Pie chart of Female survival in titanic")
plt.legend(loc='upper right', bbox_to_anchor=(2.3, 0.9),prop={'size':10})

plt.subplot(335)
labels ='Ticket class1','Ticket class2','Ticket class3'
data3 = data2["Pclass"].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.15,.15],autopct='%1.1f%%')
plt.title("Pie chart passenger based on the level")
plt.legend(loc='upper right', bbox_to_anchor=(2, 0.9),prop={'size':10})

plt.subplot(336)
labels ='class1','class2','class3'
data3 = data2["Pclass"][data2["Survived"] ==1].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.15,.15],autopct='%1.1f%%')
plt.title(" Survival rate based on the Ticket Class")
plt.legend(loc='upper right', bbox_to_anchor=(2, 0.9),prop={'size':10})

plt.subplot(337)
labels ='Survival rate','Death rate',
data3 = data2["Survived"][data2['Pclass'] ==1].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title(" Survival & Death rate: Ticket class 1")
plt.legend(loc='upper right', bbox_to_anchor=(2, 0.9),prop={'size':10})

plt.subplot(338)
labels ='Survival rate','Death rate'
data3 = data2["Survived"][data2['Pclass'] ==2].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title(" Survival & Death rate: Ticket class 2")
plt.legend(loc='upper right', bbox_to_anchor=(2, 0.9),prop={'size':10})

plt.subplot(339)
lablels ='Survival rate','Death rate'
data3 = data2["Survived"][data2['Pclass'] ==3].value_counts()
plt.pie(data3,labels=labels,explode=[0,0.01],autopct='%1.1f%%')
plt.title(" Survival & Death rate: Ticket class 3")
plt.legend(loc='upper right', bbox_to_anchor=(2, 0.9),prop={'size':10})
plt.show()