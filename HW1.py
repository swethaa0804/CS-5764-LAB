#Q1
import pandas as pd
from prettytable import PrettyTable
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = '2013-01-01'
end_date = '2023-08-28'
for i in stocks:
    print(i, 'Stocks From', start_date, 'To', end_date)
    df = data.get_data_yahoo(i, start_date, end_date)
    df = df.round(2)
    print(df.head())

#Q2
table = PrettyTable(["Name / Features", "High($)", "Low($)", "Open($)", "Close($)", "Volume", "Adj Close($)"])

table.title = "Mean Value Comparison"
high = []
low = []
open = []
close = []
volume = []
adj = []
start_date = '2013-01-01'
end_date = '2023-08-28'

for i in stocks:
    # print(i,'Stocks From',start_date,'To',end_date)
    df = data.get_data_yahoo(i, start_date, end_date)
    table.add_row([i, round(df["High"].mean(), 2), round(df["Low"].mean(), 2), round(df["Open"].mean(), 2),
                   round(df["Close"].mean(), 2), round(df["Volume"].mean(), 2), round(df["Adj Close"].mean(), 2)])
    high.append(round(df["High"].mean(), 2))
    low.append(round(df["Low"].mean(), 2))
    open.append(round(df["Open"].mean(), 2))
    close.append(round(df["Close"].mean(), 2))
    volume.append(round(df["Volume"].mean(), 2))
    adj.append(round(df["Adj Close"].mean(), 2))

table.add_row(["Maximum Value", round(max(high), 2), round(max(low), 2), round(max(open), 2), round(max(close), 2),
               round(max(volume), 2), round(max(adj), 2)])

table.add_row(["Minimum Value", round(min(high), 2), round(min(low), 2), round(min(open), 2), round(min(close), 2),
               round(min(volume), 2), round(min(adj), 2)])

table.add_row(
    ["Maximum company name", stocks[high.index(max(high))], stocks[low.index(max(low))], stocks[open.index(max(open))],
     stocks[close.index(max(close))], stocks[volume.index(max(volume))], stocks[adj.index(max(adj))]])

table.add_row(
    ["Minimum company name", stocks[high.index(min(high))], stocks[low.index(min(low))], stocks[open.index(min(open))],
     stocks[close.index(min(close))], stocks[volume.index(min(volume))], stocks[adj.index(min(adj))]])
print(table)

#Q3

table = PrettyTable(["Name / Features", "High($)", "Low($)", "Open($)", "Close($)", "Volume", "Adj Close($)"])

table.title = "Variance Comparison"

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
high = []
low = []
open = []
close = []
volume = []
adj = []
start_date = '2013-01-01'
end_date = '2023-08-28'

for i in stocks:
    # print(i,'Stocks From',start_date,'To',end_date)
    df = data.get_data_yahoo(i, start_date, end_date)
    table.add_row([i, round(df["High"].var(), 2), round(df["Low"].var(), 2), round(df["Open"].var(), 2),
                   round(df["Close"].var(), 2), round(df["Volume"].var(), 2), round(df["Adj Close"].var(), 2)])
    high.append(round(df["High"].var(), 2))
    low.append(round(df["Low"].var(), 2))
    open.append(round(df["Open"].var(), 2))
    close.append(round(df["Close"].var(), 2))
    volume.append(round(df["Volume"].var(), 2))
    adj.append(round(df["Adj Close"].var(), 2))

table.add_row(["Maximum Value", round(max(high), 2), round(max(low), 2), round(max(open), 2), round(max(close), 2),
               round(max(volume), 2), round(max(adj), 2)])

table.add_row(["Minimum Value", round(min(high), 2), round(min(low), 2), round(min(open), 2), round(min(close), 2),
               round(min(volume), 2), round(min(adj), 2)])

table.add_row(
    ["Maximum company name", stocks[high.index(max(high))], stocks[low.index(max(low))], stocks[open.index(max(open))],
     stocks[close.index(max(close))], stocks[volume.index(max(volume))], stocks[adj.index(max(adj))]])

table.add_row(
    ["Minimum company name", stocks[high.index(min(high))], stocks[low.index(min(low))], stocks[open.index(min(open))],
     stocks[close.index(min(close))], stocks[volume.index(min(volume))], stocks[adj.index(min(adj))]])
print(table)

#Q4
table = PrettyTable(["Name / Features","High($)","Low($)","Open($)","Close($)","Volume","Adj Close($)"])

table.title = "Standard Deviation Comparison"

stocks = ['AAPL','ORCL','TSLA','IBM','YELP','MSFT']
high = []
low = []
open = []
close = []
volume = []
adj = []
start_date = '2013-01-01'
end_date = '2023-08-28'

for i in stocks:

        df = data.get_data_yahoo(i,start_date,end_date)
        table.add_row([i,round(df["High"].std(),2),round(df["Low"].std(),2),round(df["Open"].std(),2),round(df["Close"].std(),2),round(df["Volume"].std(),2),round(df["Adj Close"].std(),2)])
        high.append(round(df["High"].std(),2))
        low.append(round(df["Low"].std(),2))
        open.append(round(df["Open"].std(),2))
        close.append(round(df["Close"].std(),2))
        volume.append(round(df["Volume"].std(),2))
        adj.append(round(df["Adj Close"].std(),2))

table.add_row(["Maximum Value",round(max(high),2),round(max(low),2),round(max(open),2),round(max(close),2),round(max(volume),2),round(max(adj),2)])

table.add_row(["Minimum Value",round(min(high),2),round(min(low),2),round(min(open),2),round(min(close),2),round(min(volume),2),round(min(adj),2)])

table.add_row(["Maximum company name",stocks[high.index(max(high))],stocks[low.index(max(low))],stocks[open.index(max(open))],stocks[close.index(max(close))],stocks[volume.index(max(volume))],stocks[adj.index(max(adj))]])

table.add_row(["Minimum company name",stocks[high.index(min(high))],stocks[low.index(min(low))],stocks[open.index(min(open))],stocks[close.index(min(close))],stocks[volume.index(min(volume))],stocks[adj.index(min(adj))]])
print(table)

#Q5
table = PrettyTable(["Name / Features", "High($)", "Low($)", "Open($)", "Close($)", "Volume", "Adj Close($)"])
table.title = "Median Value Comparison"

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
high = []
low = []
open = []
close = []
volume = []
adj = []
start_date = '2013-01-01'
end_date = '2023-08-28'

for i in stocks:
    # print(i,'Stocks From',start_date,'To',end_date)
    df = data.get_data_yahoo(i, start_date, end_date)
    table.add_row([i, round(df["High"].median(), 2), round(df["Low"].median(), 2), round(df["Open"].median(), 2),
                   round(df["Close"].median(), 2), round(df["Volume"].median(), 2), round(df["Adj Close"].median(), 2)])
    high.append(round(df["High"].median(), 2))
    low.append(round(df["Low"].median(), 2))
    open.append(round(df["Open"].median(), 2))
    close.append(round(df["Close"].median(), 2))
    volume.append(round(df["Volume"].median(), 2))
    adj.append(round(df["Adj Close"].median(), 2))

table.add_row(["Maximum Value", round(max(high), 2), round(max(low), 2), round(max(open), 2), round(max(close), 2),
               round(max(volume), 2), round(max(adj), 2)])

table.add_row(["Minimum Value", round(min(high), 2), round(min(low), 2), round(min(open), 2), round(min(close), 2),
               round(min(volume), 2), round(min(adj), 2)])

table.add_row(
    ["Maximum company name", stocks[high.index(max(high))], stocks[low.index(max(low))], stocks[open.index(max(open))],
     stocks[close.index(max(close))], stocks[volume.index(max(volume))], stocks[adj.index(max(adj))]])

table.add_row(
    ["Minimum company name", stocks[high.index(min(high))], stocks[low.index(min(low))], stocks[open.index(min(open))],
     stocks[close.index(min(close))], stocks[volume.index(min(volume))], stocks[adj.index(min(adj))]])
print(table)

#Q6

df = data.get_data_yahoo("AAPL", start_date, end_date)
dataframe = pd.DataFrame(df)
matrix = dataframe.corr()
matrix = matrix.round(2)
print("\t \t \tCorrelation matrix of Apple \n")
print(matrix)

#Q7

start_date = '2013-01-01'
end_date = '2023-08-28'
stocks = ['ORCL','TSLA','IBM','YELP','MSFT']

for i in stocks:
    df = data.get_data_yahoo(i,start_date,end_date)
    dataframe = pd.DataFrame(df)
    matrix =dataframe.corr()
    matrix = matrix.round(2)
    print("\t \t \t ", i, " Correlation matrix \n")
    print(matrix)

#Q8
yf.pdr_override()
start_date = '2013-01-01'
end_date = '2023-08-28'

table = PrettyTable(["Name","High($)","Low($)","Open($)","Close($)","Volume","Adj Close($)"])
table1 = PrettyTable()
table.title = "Low Standard Deviation Comparison"

stocks = ['AAPL','ORCL','TSLA','IBM','YELP','MSFT']
high = []
low = []
open = []
close = []
volume = []
adj = []
start_date = '2013-01-01'
end_date = '2023-08-28'

for i in stocks:

        df = data.get_data_yahoo(i,start_date,end_date)
        table.add_row([i,round(df["High"].std(),2),round(df["Low"].std(),2),round(df["Open"].std(),2),round(df["Close"].std(),2),round(df["Volume"].std(),2),round(df["Adj Close"].std(),2)])
        high.append(round(df["High"].std(),2))
        low.append(round(df["Low"].std(),2))
        open.append(round(df["Open"].std(),2))
        close.append(round(df["Close"].std(),2))
        volume.append(round(df["Volume"].std(),2))
        adj.append(round(df["Adj Close"].std(),2))
table.add_row(["Least Std Company",stocks[high.index(min(high))],stocks[low.index(min(low))],stocks[open.index(min(open))],stocks[close.index(min(close))],stocks[volume.index(min(volume))],stocks[adj.index(min(adj))]])
print(table)