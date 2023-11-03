import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
from io import BytesIO
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

sns.set(style='whitegrid')
# Question 1
df = pd.read_excel('Sample - Superstore.xls')
drop_features = ['Row ID', 'Order ID', 'Customer ID','Customer Name',
                 'Postal Code','Product ID','Order Date', 'Ship Date',
                 'Country', 'Segment']
df.drop(drop_features,axis=1,inplace=True)
print(df.head().round(2))

# Question 2
plt.figure(figsize=(18, 18))
plt.subplot(221)
profit = df.groupby('Category')['Profit'].sum().round(2)
profit = profit.reset_index()
label = profit.Category
profit_values = profit.Profit
plt.pie(profit_values, labels=label, explode=[0.3, 0, 0], autopct='%1.2f%%',
        textprops={'fontsize': 15})
plt.title(' Total Profit', family='serif', fontsize=35, color='blue')
plt.subplot(222)
discount = df.groupby('Category')['Discount'].sum().round(2)
discount = discount.reset_index()
label = discount.Category
discount_values = discount.Discount
plt.pie(discount_values, labels=label, explode=[0, 0, 0.3], autopct='%1.2f%%',
        textprops={'fontsize': 15})
plt.title(' Total Discount', family='serif', fontsize=35, color='blue')
plt.subplot(223)
quantity = df.groupby('Category')['Quantity'].sum().round(2)
quantity = quantity.reset_index()
label = quantity.Category
quantity_values = quantity.Quantity
plt.pie(quantity_values, labels=label, explode=[0, 0, 0.3], autopct='%1.2f%%',
        textprops={'fontsize': 15})
plt.title('Total Quantity', family='serif', fontsize=35, color='blue')
plt.subplot(224)
sales = df.groupby('Category')['Sales'].sum().round(2)
sales = sales.reset_index()
label = sales.Category
sales_values = sales.Sales
plt.pie(sales_values, labels=label, explode=[0, 0.3, 0], autopct='%1.2f%%',
        textprops={'fontsize': 15})
plt.title('Total Sales', family='serif', fontsize=35, color='blue')
plt.show()
max_profit_category = profit.loc[profit_values.idxmax()]['Category']
min_profit_category = profit.loc[profit_values.idxmin()]['Category']
max_discount_category = discount.loc[discount_values.idxmax()]['Category']
min_discount_category = discount.loc[discount_values.idxmin()]['Category']
max_quantity_category = quantity.loc[quantity_values.idxmax()]['Category']
min_quantity_category = quantity.loc[quantity_values.idxmin()]['Category']
max_sales_category = sales.loc[sales_values.idxmax()]['Category']
min_sales_category = sales.loc[sales_values.idxmin()]['Category']
print("Max Profit Category:", max_profit_category)
print("Min Profit Category:", min_profit_category)
print("Max Discount Category:", max_discount_category)
print("Min Discount Category:", min_discount_category)
print("Max Quantity Category:", max_quantity_category)
print("Min Quantity Category:", min_quantity_category)
print("Max Sales Category:", max_sales_category)
print("Min Sales Category:", min_sales_category)

# Question 3
data = pd.DataFrame(df)
table = PrettyTable(["","Sales ($)","Quantity","Discounts ($)","Profit ($)"])
column = ["Sales","Quantity","Discount","Discounts"]
sales = data.groupby('Category')['Sales'].sum().round(2)
sales = sales.reset_index()
quantity = data.groupby('Category')['Quantity'].sum().round(2)
quantity = quantity.reset_index()
discount = data.groupby('Category')['Discount'].sum().round(2)
discount = discount.reset_index()
profit = data.groupby('Category')['Profit'].sum().round(2)
profit = profit.reset_index()
table.title = 'Super store - Category'
table.add_row(["'Furniture'",sales['Sales'][0],quantity['Quantity'][0],
               discount['Discount'][0],profit['Profit'][0]])
table.add_row(["'Office Supplies'",sales['Sales'][1],quantity['Quantity'][1],
               discount['Discount'][1],profit['Profit'][1]])
table.add_row(["'Technology'",sales['Sales'][2],quantity['Quantity'][2],
               discount['Discount'][2],profit['Profit'][2]])
table.add_row(["Maximum Value",max(sales['Sales']),max(quantity['Quantity']),
               max(discount['Discount']),max(profit['Profit'])])
table.add_row(["Maximum Value",min(sales['Sales']),min(quantity['Quantity']),
               min(discount['Discount']),min(profit['Profit'])])
table.add_row(["Maximum Feature",'Technology','Office Supplies','Office Supplies',
               'Technology'])
table.add_row(["Minimum Feature",'Office Supplies','Technology','Technology','Furniture'])
print(table)

# Question 4
sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum().reset_index()
profit_by_subcategory = df.groupby('Sub-Category')['Profit'].sum().reset_index()
subcategories_order = ['Phones', 'Chairs', 'Storage', 'Tables', 'Binders',
                       'Machines', 'Accessories', 'Copiers', 'Bookcases', 'Appliances']
filtered_sales = sales_by_subcategory.set_index('Sub-Category').loc[subcategories_order].reset_index()
filtered_profit = profit_by_subcategory.set_index('Sub-Category').loc[subcategories_order].reset_index()
fig, ax = plt.subplots(figsize=(20, 8))
bars = ax.bar(filtered_sales['Sub-Category'], filtered_sales['Sales'],
              width=0.4, color='#95DEE3', edgecolor='blue', label='Sales')
ax.set_xlabel('Sales', fontsize=25)
ax.set_ylabel('USD ($)', fontsize=25)
bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
ax.set_xticks(bar_positions)
ax.tick_params(axis='both', labelsize=20)
ax.set_xticklabels(filtered_sales['Sub-Category'], fontsize=20, ha='center')
for k, v in filtered_sales["Sales"].astype("float").round(2).items():
    if v > 300000:
        plt.text(
            k,
            v - 90000,
            "$" + str(v),
            fontsize=20,
            color="k",
            horizontalalignment="center",
            rotation="vertical",
            verticalalignment="bottom",
        )
    else:
        plt.text(
            k,
            v + 10000,
            "$" + str(v),
            fontsize=20,
            color="k",
            horizontalalignment="center",
            rotation="vertical",
            verticalalignment="bottom",
        )

ax2 = ax.twinx()
ax2.plot(filtered_profit['Sub-Category'], filtered_profit['Profit'], color='red', linewidth=4, label='Profit', marker='o')
ax2.set_ylabel('USD ($)', fontsize=25)
ax2.tick_params(axis='y', labelsize=20)
y_limit = (-50000, 350000)
ax.set_ylim(y_limit)
ax2.set_ylim(y_limit)
ax.grid(axis='both', linestyle='--', alpha=0.6)
fig.suptitle('Sales and Profit per Sub-Category', fontsize=30)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
all_handles = handles1 + handles2
all_labels = labels1 + labels2
ax.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(1, 1.1130),
          fontsize=15, frameon=False)
plt.show()

# Question 5
sns.set(style='whitegrid')
x = np.linspace(0,2* np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
fig, ax = plt.subplots()
line_width = 3
ax.plot(x, y_sin, lw = line_width, label='Sine Wave', linestyle='--')
ax.plot(x,y_cos, lw=line_width, label = 'Cosine Wave', linestyle = 'dashdot')
ax.fill_between(x, y_sin, y_cos, where=(y_sin >= y_cos), color='green',
                alpha=0.3, interpolate=True)
ax.fill_between(x, y_sin, y_cos, where=(y_sin < y_cos), color='orange',
                alpha=0.3, interpolate=True)
title_style = {'family':'serif', 'color': 'darkred', 'size':15}
label_style = {'family': 'serif', 'color': 'darkred', 'size':15}
ax.set_xlabel('x-axis', fontdict=label_style)
ax.set_ylabel('y-axis', fontdict=label_style)
plt.title("Fill between x-axis and plot line",
          fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
ax.legend(fontsize=15)
ax.grid()
plt.show()

# Question 6
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-4, 4, 800)
y = np.linspace(-4, 4, 800)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0,
                          antialiased=False, alpha=1)
ax.set_zlim(-5, 2)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.contour(X, Y, Z, zdir='x', offset=-5, cmap='coolwarm', linewidths=1)
ax.contour(X, Y, Z, zdir='y', offset=5, cmap='coolwarm', linewidths=1)
ax.contour(X, Y, Z, zdir='z', offset=-5, cmap='coolwarm', linewidths=1)
ax.set_xlabel("X Label", fontsize=15, family="serif", color="darkred")
ax.set_ylabel("Y Label", fontsize=15, family="serif", color="darkred")
ax.set_zlabel("Z Label", fontsize=15, family="serif", color="darkred")
ax.set_title(r'surface plot of z= sin $\sqrt{x^2 + y^2}$',
             fontdict={'family': 'serif', 'color': 'blue', 'size': 25})
plt.show()



# Question 7
sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum().reset_index()
profit_per_subc = df.groupby('Sub-Category')['Profit'].sum().reset_index()

subcategories_order = ['Phones', 'Chairs', 'Storage', 'Tables', 'Binders',
                       'Machines', 'Accessories', 'Copiers', 'Bookcases', 'Appliances']

sales_by_subcategory = sales_by_subcategory.set_index('Sub-Category').loc[subcategories_order].reset_index()
desired_subcategories = ['Phones', 'Chairs', 'Storage', 'Tables', 'Binders',
                         'Machines', 'Accessories', 'Copiers', 'Bookcases', 'Appliances']

profit_per_subc = profit_per_subc.set_index('Sub-Category').loc[desired_subcategories].reset_index()

sorted_df_sales = sales_by_subcategory[sales_by_subcategory['Sub-Category'].isin(desired_subcategories)]
sorted_df_profit = profit_per_subc[profit_per_subc['Sub-Category'].isin(desired_subcategories)]
fig = plt.figure(figsize=(9, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
ax0 = plt.subplot(gs[0, :])
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[1, 1])
x = np.arange(0, len(desired_subcategories))
width = 0.4
bar_sales = ax0.bar(x - width/2, sorted_df_sales['Sales'], width, label='Sales', color='#95DEE3',
                    edgecolor='blue')
ax0.bar(x + width/2, sorted_df_profit['Profit'], width, label='Profit', color='lightcoral',
        edgecolor='red')
ax0.set_ylim(-50000, 350000)
ax0.set_xlabel("Sales", fontsize=10)
ax0.set_ylabel("USD($)", fontsize=10)
ax0.set_title("Sales and Profit per sub-category", fontsize=15)
bar_positions = [bar.get_x()+ bar.get_width()/2 for bar in bar_sales]
ax0.set_xticks(bar_positions)
ax0.tick_params(axis='both',labelsize=10)
ax0.set_xticklabels(desired_subcategories,fontsize=10,ha='center')
ax0.legend(fontsize=10,frameon=False)
ax0.grid(axis='both', linewidth=0.7, alpha=0.7)
sales_category = df.groupby('Category')['Sales'].sum()
ax1.pie(sales_category, labels=sales_category.index, autopct='%1.2f%%')
ax1.set_title('Sales', fontsize=15)
profit_category = df.groupby('Category')['Profit'].sum()
ax2.pie(profit_category, labels=profit_category.index, autopct='%1.2f%%')
ax2.set_title('Profit', fontsize=15)
plt.tight_layout()
plt.show()