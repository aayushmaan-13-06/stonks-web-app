import numpy as np
import pandas as pd
import re
import seaborn as sns
from pandas_datareader import data
from pandas_datareader.data import DataReader
from datetime import datetime
from scipy import stats
import statistics
import matplotlib.pyplot as plt
from scipy.stats import skew
import streamlit as st
import plotly.express as ff
import webbrowser
st.write("""

## Prepared by:
## Aayushmaan Jain
""")
st.write("""
This is a web app intended to help you with basic stock analysis along with the visulalizations with a statistical perspective.

Precautions and advices

1) This notebook uses Yahoo Finance as its source of financial data by default

2) The case in which you enter the stock does not matter as the stock name will be converted to upper case

3) The separator you use while entering the dates does not matter as the date accepts both separators like / and - but please enter the date in YYYY/MM/DD format

Please note that this web app only helps you to analyse the stocks and take informed decisions. This app does not give any financial advice. Please invest carefully.
""")

np.set_printoptions(threshold=np.inf) #displaying full numpy array without truncation
pd.set_option("display.max_rows", None, "display.max_columns", None) #displaying full pandas dataframe without truncation

st.write('# User Inputs')
st.write('### Enter quit to stop')
stock_list = list()
num_stocks = st.number_input('Enter the number of stocks in your portfolio')
key = 1
for i in range(int(num_stocks)):
    stock_inp = st.text_input('Enter a stock name according to yahoo finance search: ', key = str(key))
    stock_list.append(stock_inp.upper())
    key += 1
from_date = st.text_input('Enter the from date in yyyy/mm/dd format ')
from_date = re.split('-|/', from_date)
to_date = st.text_input('Enter the to date in yyyy/mm/dd format ')
to_date = re.split('-|/', to_date)
try:
    for i in range(len(from_date)):
      from_date[i] = int(from_date[i])
    for j in range(len(to_date)):
      to_date[j] = int(to_date[j])
except:
    pass

column_name = st.text_input('Enter the column you would like to consider for analysis (case sensitive): ')
final_dataframe = pd.DataFrame()
try:
  for stock in stock_list:
    historical_data = DataReader(stock,  "yahoo", datetime(int(from_date[0]), int(from_date[1]), int(from_date[2])), datetime(int(to_date[0]), int(to_date[1]), int(to_date[2])))
    final_dataframe[stock] = historical_data[column_name]
except:
  st.write('Please enter a valid input')
final_dataframe = final_dataframe.dropna()

return_dataframe = final_dataframe.pct_change()
return_dataframe = return_dataframe.dropna()
try:
    return_dataframe.columns = stock_list
except:
    pass

st.write("""
This web app can analyse the stocks on the basis of following plots

1) Histogram

2) Boxplots

3) Time Series

4) Comparitive Analysis
""")
choice_hist = st.text_input('Would you like to analyse based on Histograms? ')
choice_box = st.text_input('Would you like to analyse based on Boxplots? ')
choice_ts = st.text_input('Would you like to analyse based on Time Series? ')
choice_comparitive = st.text_input('Would you like to do a comparitive analysis? ')
choice_col = st.text_input(f'Would you like to analyse returns or {column_name} or both? ')
try:
    if choice_hist.upper().startswith('Y') or choice_hist.upper().startswith('B'):
        if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
            st.write('Histograms for returns')
            for stock in stock_list:
              title = f'Histogram for returns of {stock}'
              fig = ff.histogram(return_dataframe, x=stock, opacity = 0.5, title = title)
              st.plotly_chart(fig)
              st.write('The histogram shows that the returns of', stock, 'oscillate around', stats.mode(return_dataframe[stock])[0][0])
              st.write('The average return of this stock is', np.mean(return_dataframe[stock]))
              st.write('The median return of this stock is', np.median(return_dataframe[stock]))
              if return_dataframe[stock].skew()>0:
                st.write('This stock is positively skewed which indicates that there are more positive returns which is a good sign for the stock')
                st.write('This is supported by the more number of peaks and higher peaks on the positive side of the histogram after the meidan')
              elif return_dataframe[stock].skew()<0:
                st.write('This stock is negatively skewed which indicates that there are more negative returns which is a bad sign for the stock')
                st.write('This is supported by the more number of peaks and higher peaks on the negative side of the histogram before the meidan')
              else:
                st.write('The stock is not skewed which means both the returns are equally likely')
              st.write('Please look out for the outliers also')
        if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
            st.write('Histograms for {col}'.format(col=column_name))
            for stock in stock_list:
              title = f'Histogram for {column_name} of {stock}'
              fig = ff.histogram(final_dataframe, x=stock, opacity = 0.5, title = title)
              st.plotly_chart(fig)
              st.write('The histogram shows that the',column_name,'of', stock, 'oscillate around', stats.mode(final_dataframe[stock])[0][0])
              st.write('The average ',column_name,' of this stock is', np.mean(final_dataframe[stock]))
              st.write('The median ',column_name,' of this stock is', np.median(final_dataframe[stock]))
              if final_dataframe[stock].skew()>0:
                st.write('This stock is positively skewed which indicates that there are more positive {col} which is a good sign for the stock'.format(col=column_name))
                st.write('This is supported by the more number of peaks and higher peaks on the positive side of the histogram after the meidan')
              elif final_dataframe[stock].skew()<0:
                st.write('This stock is negatively skewed which indicates that there are more negative {col} which is a bad sign for the stock'.format(col=column_name))
                st.write('This is supported by the more number of peaks and higher peaks on the negative side of the histogram before the meidan')
              else:
                st.write('The stock is not skewed which means both the {col} are equally likely'.format(col=column_name))
              st.write('Please look out for the outliers also')

    if choice_box.upper().startswith('Y') or choice_box.upper().startswith('B'):
        if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
            st.write('Boxplots for retuns')
            for stock in stock_list:
              title = f'Boxplots for returns of stock {stock}'
              fig = ff.box(return_dataframe, y=stock, title = title)
              st.plotly_chart(fig)
              st.write('From this boxplot we can see that the median retruns of this stock', stock, 'is ', np.median(return_dataframe[stock]))
              if return_dataframe[stock].skew()>0:
                st.write('From this boxplot we can see that there are more positive returns in this stock')
                st.write('This can be seen in the boxplot as there are more returns above the median i.e there is more data above the median than below the median')
              elif return_dataframe[stock].skew()<0:
                st.write('From this boxplot we can see that there are more negative returns in this stock')
                st.write('This can be seen in the boxplot as there are more returns below the median i.e there is more data below the median than above the median')
              else:
                st.write('The returns are not skewed and the positive returns and the negative returns are equally likely')
              st.write('Please watch out for the outliers')
        if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
            st.write('Boxplot for {col}'.format(col = column_name))
            for stock in stock_list:
              title = f'Boxplots for {column_name} of stock {stock}'
              fig = ff.box(final_dataframe, y=stock, title = title)
              st.plotly_chart(fig)
              st.write('From this boxplot we can see that the median',column_name ,'of this stock', stock, 'is ', np.median(final_dataframe[stock]))
              if final_dataframe[stock].skew()>0:
                st.write('From this boxplot we can see that there are more positive {col} in this stock'.format(col=column_name))
                st.write('This can be seen in the boxplot as there are more {col} above the median i.e there is more data above the median than below the median'.format(col=column_name))
              elif final_dataframe[stock].skew()<0:
                st.write('From this boxplot we can see that there are more negative {col} in this stock'.format(col=column_name))
                st.write('This can be seen in the boxplot as there are more {col} below the median i.e there is more data below the median than above the median'.format(col = column_name))
              else:
                st.write('The returns are not skewed and the positive {col} and the negative returns are equally likely'.format(col=column_name))
              st.write('Please watch out for the outliers')

    if choice_ts.upper().startswith('Y') or choice_ts.upper().startswith('B'):
        if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
            st.write('Time Series plot for returns')
            for stock in stock_list:
              title = 'The time series plot for the returns of the stock {stock}'.format(stock = stock)
              fig = ff.line(return_dataframe, x=return_dataframe.index, y = stock, title=title)
              st.plotly_chart(fig)
              st.write('The maximum return of this stock was observed on {max}'.format(max = return_dataframe[stock].idxmax()))
              st.write('The minimum return of this stock was observed on {min}'.format(min = return_dataframe[stock].idxmin()))
        if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
            st.write('Time Series plots for {col}'.format(col = column_name))
            for stock in stock_list:
              title = f'The time series plot for the {column_name} of the stock {stock}'
              fig = ff.line(final_dataframe, x=final_dataframe.index, y=stock, title = title)
              st.plotly_chart(fig)
              st.write(f'The maximum {column_name} of this stock was observed on {final_dataframe[stock].idxmax()}')
              st.write(f'The minimum {column_name} of this stock was observed on {final_dataframe[stock].idxmin()}')

    """Descriptive Statistics"""

    mean_list_returns = []
    median_list_returns = []
    mode_list_returns = []
    skewness_list_returns = []
    kurtosis_list_returns = []
    stdev_list_returns = []
    mean_list_column = []
    median_list_column = []
    mode_list_column = []
    skewness_list_column = []
    kurtosis_list_column = []
    stdev_list_column = []
    sharpe_ratio_return = []
    min_return = []
    max_return = []
    min_col = []
    max_col = []
    risk_free_rate = 0.06
    for stock in stock_list:
      mean_list_returns.append(np.mean(return_dataframe[stock]))
      median_list_returns.append(np.median(return_dataframe[stock]))
      mode_list_returns.append(stats.mode(return_dataframe[stock])[0][0])
      skewness_list_returns.append(return_dataframe[stock].skew())
      kurtosis_list_returns.append(stats.kurtosis(return_dataframe[stock]))
      stdev_list_returns.append(np.std(return_dataframe[stock]))
      mean_list_column.append(np.mean(final_dataframe[stock]))
      median_list_column.append(np.median(final_dataframe[stock]))
      mode_list_column.append(stats.mode(final_dataframe[stock])[0][0])
      skewness_list_column.append(final_dataframe[stock].skew())
      kurtosis_list_column.append(stats.kurtosis(final_dataframe[stock]))
      stdev_list_column.append(np.std(final_dataframe[stock]))
      sharpe_ratio_return.append((np.mean(return_dataframe[stock])-risk_free_rate)/np.std(return_dataframe[stock]))
      min_return.append(np.min(return_dataframe[stock]))
      min_col.append(np.min(final_dataframe[stock]))
      max_return.append(np.max(return_dataframe[stock]))
      max_col.append(np.max(final_dataframe[stock]))

    desc_stats_returns = pd.DataFrame(index = range(len(stock_list)), columns=range(9))
    desc_stats_returns.columns = ['Mean', 'Median', 'Mode', 'Stdev', 'Skewness', 'Kurtosis', 'Sharpe Ratio','Max', 'Min']
    desc_stats_returns.index = stock_list
    if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
        st.write('Descriptive Statistics Table for Returns')
        for stock in stock_list:
          desc_stats_returns['Mean'] = mean_list_returns
          desc_stats_returns['Median']  = median_list_returns
          desc_stats_returns['Mode'] = mode_list_returns
          desc_stats_returns['Stdev'] = stdev_list_returns
          desc_stats_returns['Skewness'] = skewness_list_returns
          desc_stats_returns['Kurtosis'] = kurtosis_list_returns
          desc_stats_returns['Sharpe Ratio'] = sharpe_ratio_return
          desc_stats_returns['Max'] = max_return
          desc_stats_returns['Min'] = min_return
        desc_stats_returns.index.name = 'Stocks'
        st.dataframe(desc_stats_returns)

    desc_stats_column = pd.DataFrame(index = range(len(stock_list)), columns=range(8))
    desc_stats_column.columns = ['Mean', 'Median', 'Mode', 'Stdev', 'Skewness', 'Kurtosis', 'Min', 'Max']
    desc_stats_column.index = stock_list
    if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
        st.write('Descriptive Statistics Table for', column_name)
        for stock in stock_list:
          desc_stats_column['Mean'] = mean_list_column
          desc_stats_column['Median']  = median_list_column
          desc_stats_column['Mode'] = mode_list_column
          desc_stats_column['Stdev'] = stdev_list_column
          desc_stats_column['Skewness'] = skewness_list_column
          desc_stats_column['Kurtosis'] = kurtosis_list_column
          desc_stats_column['Max'] = max_col
          desc_stats_column['Min'] = min_col
        desc_stats_column.index.name = 'Stocks'
        st.dataframe(desc_stats_column)

    if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
        st.write('Correlation Matirix for Retruns')
        correlation_matrix = return_dataframe.corr()
        correlation_matrix.index.name = 'Stocks'
        st.dataframe(correlation_matrix)

    if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
        st.write('Correlation Matirix for', column_name)
        correlation_matrix_col = final_dataframe.corr()
        correlation_matrix_col.index.name = 'Stocks'
        st.dataframe(correlation_matrix_col)

    st.write('Visualizing correlation martrix')
    if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
        fig = plt.figure(figsize=(final_dataframe.shape[1],final_dataframe.shape[1]))
        sns.heatmap(correlation_matrix, annot=True, fmt ='.3g', cmap='coolwarm', square=True).set_title('Correlation Matrix for Returns')
        st.pyplot(fig)

    if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
        st.write('Correlation matrix for {col}'.format(col = column_name))
        fig = plt.figure(figsize=(final_dataframe.shape[1],final_dataframe.shape[1]))
        sns.heatmap(correlation_matrix_col, annot=True, fmt ='.3g', cmap='coolwarm', square=True).set_title('Correlation Matrix for {column}'.format(column = column_name))
        st.pyplot(fig)

    if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
        st.write('Covariance Matrix')
        covMatrix = pd.DataFrame.cov(return_dataframe)
        covMatrix.index.name = 'Stocks'
        st.dataframe(covMatrix)

    if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
        st.write('Covariance Matrix for', column_name)
        covMatrix_col = pd.DataFrame.cov(final_dataframe)
        covMatrix_col.index.name = 'Stocks'
        st.dataframe(covMatrix_col)

    st.write('Visualizing covariance matrix ')
    if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
        fig = plt.figure(figsize=(final_dataframe.shape[1],final_dataframe.shape[1]))
        sns.heatmap(covMatrix, annot=True, cmap='icefire', square=True).set_title('Covariance Matrix for Returns')
        st.pyplot(fig)

    if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
        st.write('Covariance matrix for {col}'.format(col = column_name))
        fig = plt.figure(figsize=(final_dataframe.shape[1],final_dataframe.shape[1]))
        sns.heatmap(covMatrix_col, annot=True, cmap='icefire', square=True).set_title('Covariance Matrix for {column}'.format(column = column_name))
        st.pyplot(fig)

    if choice_comparitive.upper().startswith('Y'):
        st.write('Comparitive Analysis')
        if choice_col.upper().startswith('R') or choice_col.upper().startswith('B'):
            st.line_chart(return_dataframe)
            most_vol_stock_returns = stock_list[stdev_list_returns.index(np.max(stdev_list_returns))]
            least_vol_stock_returns = stock_list[stdev_list_returns.index(np.min(stdev_list_returns))]
            most_skew_stock_returns = stock_list[skewness_list_returns.index(np.max(skewness_list_returns))]
            least_skew_stock_returns = stock_list[skewness_list_returns.index(np.min(skewness_list_returns))]
            most_avg_returns = stock_list[mean_list_returns.index(np.max(mean_list_returns))]
            least_avg_returns = stock_list[mean_list_returns.index(np.min(mean_list_returns))]
            st.write(f'Most Volatile stock is:{most_vol_stock_returns}')
            st.write(f'Least Volatile stock is:{least_vol_stock_returns}')
            st.write(f'Most Skewed stock is: {most_skew_stock_returns}')
            st.write(f'Least Skewed stock is: {least_skew_stock_returns}')
            st.write(f'{most_avg_returns} has the most average returns')
            st.write(f'{least_avg_returns} has the least average returns')
        if choice_col.upper().startswith(column_name.upper()) or choice_col.upper().startswith('B'):
            st.line_chart(final_dataframe)
            most_vol_stock_col = stock_list[stdev_list_column.index(np.max(stdev_list_column))]
            least_vol_stock_col = stock_list[stdev_list_column.index(np.min(stdev_list_column))]
            most_skew_stock_col = stock_list[skewness_list_column.index(np.max(skewness_list_column))]
            least_skew_stock_col = stock_list[skewness_list_column.index(np.min(skewness_list_column))]
            most_avg_column = stock_list[mean_list_column.index(np.max(mean_list_column))]
            least_avg_column = stock_list[mean_list_column.index(np.min(mean_list_column))]
            st.write(f'Most Volatile stock is:{most_vol_stock_col}')
            st.write(f'Least Volatile stock is: {least_vol_stock_col}')
            st.write(f'Most Skewed stock is:{most_skew_stock_col}')
            st.write(f'Least Skewed stock is: {least_skew_stock_col}')
            st.write(f'{most_avg_column} has the most average {column_name}')
            st.write(f'{least_avg_column} has the least average {column_name}')
except:
    pass
st.write("""
Thank you for using this web app, please leave a feedback. This motivates me to make more web apps like this.

Regards,

Aayushmaan Jain
""")

url = 'https://aayush1036.github.io/profile_website/'
if st.button('See my website'):
    webbrowser.open_new_tab(url)
