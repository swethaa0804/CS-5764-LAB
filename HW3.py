import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

np.random.seed(5764)
warnings.filterwarnings("ignore")

# Question 1
df = sns.load_dataset('penguins')
print('The last five observations:')
print(df.tail().round(2))
print('The data statistics:')
print(df.describe().round(2))

# Question 2
print('Null values before dropping: ', df.isnull().sum().sum().round(2))
df_cleaned = df.dropna()
print('Null values after dropping: ', df_cleaned.isnull().sum().sum().round(2))
if df_cleaned.isna().any().any():
    print("Dataset still has missing values!")
else:
    print("Dataset is now clean!")
print("Cleaned Dataset:")
print(df_cleaned)


# Question 3
sns.set(style="darkgrid")
sns.histplot(data=df_cleaned, x='flipper_length_mm', kde=True)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Count')
plt.title('Question 3')
plt.show()

# Question 4
sns.set(style="darkgrid")
sns.histplot(data=df_cleaned, x='flipper_length_mm', kde=True,binwidth=3)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Count')
plt.title('Question 4')
plt.show()

# Question 5
sns.set(style="darkgrid")
sns.histplot(data=df_cleaned, x='flipper_length_mm', kde=True,bins=30)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Count')
plt.title('Question 5')
plt.show()

# Question 6
sns.set(style="darkgrid")
sns.displot(data=df_cleaned, x='flipper_length_mm', hue='species', kde=True)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Count')
plt.title('Question 6',y=0.99)
plt.show()

# Question 7
plt.figure()
sns.displot(data=df, x="flipper_length_mm", hue="species", kde=True, element="step")
plt.xlabel("flipper_length_mm")
plt.ylabel("frequency")
plt.title("Question 7")
plt.grid(True)
plt.tight_layout()
plt.show()

# Show the plot
plt.show()

# Question 8
sns.set(style="darkgrid")
sns.histplot(data=df_cleaned, x='flipper_length_mm', hue='species', multiple='stack')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Count')
plt.title('Question 8',y=0.99)
plt.show()

# Question 9
sns.set(style="darkgrid")
sns.displot(data=df_cleaned, x='flipper_length_mm', hue='sex', multiple='dodge')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Count')
plt.title('Question 9',y=0.99)
plt.show()


# Question 10

sns.displot(data=df_cleaned, x="flipper_length_mm", col="sex",hue="sex")
plt.xlabel("flipper_length_mm")
plt.ylabel("frequency")
plt.suptitle('Question 10')
plt.grid(True)
plt.tight_layout()
plt.show()

#Question 11

sns.histplot(data=df_cleaned, x="flipper_length_mm", hue="species", stat="density")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 11",y=0.99)
plt.show()

#Question 12


sns.set(style="darkgrid")
g = sns.histplot(data=df_cleaned, x='flipper_length_mm', hue='sex', kde=True, stat="density")
# Set the title
plt.title('Question 12',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()


#Question 13

sns.set(style="darkgrid")

g = sns.displot(data=df_cleaned, x='flipper_length_mm', hue='species',  stat='probability',kde=True)


plt.title('Question 13',y=0.99)

plt.xlabel('Flipper Length (mm)')
plt.ylabel('Probability')
plt.show()

#Question 14
sns.set(style="darkgrid")
g = sns.displot(data=df_cleaned, x='flipper_length_mm', hue='species', kind='kde')
plt.title('Question 14',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()

#Question 15
g = sns.displot(data=df_cleaned, x='flipper_length_mm', hue='sex', kind='kde')
plt.title('Question 15',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()

#Question 16
sns.set(style="darkgrid")
g = sns.displot(data=df_cleaned, x='flipper_length_mm', hue='species', kind='kde',multiple='stack')
plt.title('Question 16',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()

#Question 17
sns.set(style="darkgrid")
g = sns.displot(data=df_cleaned, x='flipper_length_mm', hue='sex', kind='kde',multiple='stack')
plt.title('Question 17',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()

#Question 18,Need Observation
sns.set(style="darkgrid")
g = sns.displot(data=df_cleaned, x='flipper_length_mm',hue='species', kind='kde',fill= True)
plt.title('Question 18',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()


#Question 19,Need Observation
sns.set(style="darkgrid")
g = sns.displot(data=df_cleaned, x='flipper_length_mm', hue='sex', kind='kde',fill= True)
plt.title('Question 19',y=0.99)
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()

#Question 20,Need Observation
sns.set(style="darkgrid")
sns.lmplot(data=df_cleaned, x='bill_length_mm', y='bill_depth_mm')
plt.title('Question 20', y=1.05)
print('Correlation between bill_length_mm and bill_depth_mm: ',
      df_cleaned['bill_length_mm'].corr(df_cleaned['bill_depth_mm']).round(2))
plt.tight_layout()
plt.show()
#Question 21,Need Observation
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='island', hue='species')
plt.title('Question 21',y=0.99)
plt.xlabel('Island')
plt.ylabel('Count')
plt.legend(title='Species', loc='upper right')
plt.show()

#Question 22,Need Observation
sns.set(style="darkgrid")
sns.countplot(data=df_cleaned, x='sex', hue='species')
plt.title('Question 22',y=0.99)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Species', loc='upper right')
plt.show()

#Question 23
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_cleaned, x='bill_length_mm', y='bill_depth_mm', hue='sex', shade=True)
plt.title('Question 23',y=0.99)
plt.xlabel('Bill_length_mm')
plt.ylabel('Bill_Depth_mm')
plt.show()
print('Correlation between bill_length_mm and bill_depth_mm: ',
      df_cleaned['bill_length_mm'].corr(df_cleaned['bill_depth_mm']).round(2))

#Question 24
sns.set(style="darkgrid")
sns.kdeplot(data=df_cleaned, x='bill_length_mm', y='flipper_length_mm', hue='sex', shade=True)
plt.title('Question 24',y=0.99)
plt.xlabel('Bill_length_mm')
plt.ylabel('Flipper_Length_mm')
plt.show()
print('Correlation between bill_length_mm and flipper_length_mm: ',
      df_cleaned['bill_length_mm'].corr(df_cleaned['flipper_length_mm']).round(2))

#Question 25
sns.set(style="darkgrid")
sns.kdeplot(data=df_cleaned, x='flipper_length_mm', y='bill_depth_mm', hue='sex', shade=True)
plt.xlabel('flipper_length_mm')
plt.ylabel('bill_depth_mm')
plt.title('Question 25',y=0.99)
plt.show()
print('Correlation between flipper_length_mm and bill_depth_mm : ',
        df_cleaned['flipper_length_mm'].corr(df_cleaned['bill_depth_mm']).round(2))

#Question 26

sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
sns.kdeplot(data=df_cleaned, x="bill_length_mm", y="bill_depth_mm", hue="sex", fill=True, ax=axes[0])
sns.kdeplot(data=df_cleaned, x="bill_length_mm", y="flipper_length_mm", hue="sex", fill=True, ax=axes[1])
sns.kdeplot(data= df_cleaned, x="bill_depth_mm", y="flipper_length_mm", hue="sex", fill=True, ax=axes[2])
plt.suptitle('Question 26')
plt.grid(True)
plt.tight_layout()
plt.show()

#Question 27
sns.set(style="darkgrid")
sns.histplot(data=df_cleaned, x='bill_length_mm', y='bill_depth_mm', hue='sex')
plt.title('Question 27',y=0.99)
plt.xlabel('Bill_length_mm')
plt.ylabel('Bill_Depth_mm')
plt.show()

#Question 28
sns.set(style="darkgrid")
sns.displot(data=df_cleaned, x='bill_length_mm', y='flipper_length_mm', hue='sex')
plt.title('Question 28',y=0.99)
plt.xlabel('Bill_length_mm')
plt.ylabel('Flipper_Length_mm')
plt.show()

#Question 29
sns.set(style="darkgrid")
sns.displot(data=df_cleaned, y='flipper_length_mm', x='bill_depth_mm', hue='sex')
plt.xlabel('flipper_length_mm')
plt.ylabel('bill_depth_mm')
plt.title('Question 29',y=0.99)
plt.show()
