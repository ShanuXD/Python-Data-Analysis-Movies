import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

pd.set_option('display.width', 320)
pd.set_option('display.max.columns', 10)
pd.options.display.float_format = '{:,.2f}'.format
cost_revenue_df = pd.read_csv('cost_revenue_dirty.csv')
print(cost_revenue_df.shape)
# print(cost_revenue_df)

"""Check for NaN value and duplicate"""
# print(cost_revenue_df.columns.isna())

# print(cost_revenue_df.isna().values.any())
# print(cost_revenue_df.duplicated().values.any())

"""Check Data Type"""
# print(cost_revenue_df.info())

"""Remove $ , From Dataset And Covert It Into int and datetime Type"""
char_to_remove = ['$', ',']
cols_to_clean = ['USD_Production_Budget', 'USD_Worldwide_Gross', 'USD_Domestic_Gross']

for col in cols_to_clean:
    for ch in char_to_remove:
        cost_revenue_df[col] = cost_revenue_df[col].astype(str).str.replace(ch, '', regex=True)
    cost_revenue_df[col] = pd.to_numeric(cost_revenue_df[col])

cost_revenue_df.Release_Date = pd.to_datetime(cost_revenue_df['Release_Date'])
# print(cost_revenue_df.info())

"""Avg Production Budget Of A Movie in Dataset"""
# print(cost_revenue_df['USD_Production_Budget'].mean())
# or use .describe method
# print(cost_revenue_df.describe())

# print(cost_revenue_df[cost_revenue_df.USD_Production_Budget == 1100.00])

zero_domestic = cost_revenue_df[cost_revenue_df['USD_Domestic_Gross'] == 0]
# print(zero_domestic)
zero_domestic.sort_values('USD_Production_Budget', ascending=False)
# print(zero_domestic)

zero_worldwide = cost_revenue_df[cost_revenue_df['USD_Worldwide_Gross'] == 0]
zero_worldwide = zero_worldwide.sort_values('USD_Production_Budget', ascending=False)
# print(zero_worldwide)


"""Filter on Multiple Conditions: International Films"""
# which films made money internationally
international_releases = cost_revenue_df.loc[(cost_revenue_df.USD_Domestic_Gross == 0) & (cost_revenue_df.USD_Worldwide_Gross != 0)]
international_releases = cost_revenue_df.query('USD_Domestic_Gross == 0 and USD_Worldwide_Gross != 0')
# print(international_releases)

"""Remove Unreleased Films"""
scrape_date = pd.Timestamp('2018-5-1')
future_release = cost_revenue_df[cost_revenue_df['Release_Date'] >= scrape_date]
# print(future_release)
clean_df = cost_revenue_df.drop(future_release.index)
# print(clean_df)
# money_losing = clean_df.loc[clean_df.USD_Production_Budget > clean_df.USD_Worldwide_Gross]
# print(len(money_losing)/len(clean_df))

# Using .query()
money_losing = clean_df.query('USD_Production_Budget > USD_Worldwide_Gross')
# print(money_losing)
# print(money_losing.shape[0]/clean_df.shape[0])

"""Data Visualisation"""
# Seaborn Scatter Plots
# Revenue Vs Budget

# plt.figure(figsize=(6, 4), dpi=200)
# ax = sns.scatterplot(data=clean_df,
#                     x='USD_Production_Budget',
#                     y='USD_Worldwide_Gross',
#                     hue='USD_Worldwide_Gross',  # colour
#                     size='USD_Worldwide_Gross',  # dot size
#                      )
#
# ax.set(ylim=(0, 3000000000),
#        xlim=(0, 450000000),
#        ylabel='Revenue in $ billions',
#        xlabel='Budget in $100 millions')
# plt.show()

"""Budget Vs Year"""
# budget_in_million_df = clean_df.query('USD_Production_Budget <= 100000000')
# # print(budget_in_million_df)
# plt.figure(figsize=(6, 4), dpi=200)
# with sns.axes_style("whitegrid"):
#     ax = sns.scatterplot(data=clean_df,
#                          x='Release_Date',
#                          y='USD_Production_Budget',
#                          hue='USD_Worldwide_Gross',
#                          size='USD_Worldwide_Gross', )
#
#     ax.set(ylim=(0, 450000000),
#            xlim=(clean_df.Release_Date.min(), clean_df.Release_Date.max()),
#            xlabel='Year',
#            ylabel='Budget in $100 millions')
#     plt.show()

dt_index = pd.DatetimeIndex(clean_df['Release_Date'])
years = dt_index.year
# print(years)
decades = (years//10)*10
clean_df['Decade'] = decades
# print(clean_df)

old_films_df = clean_df[clean_df['Decade'] < 1970]
new_flims_df = clean_df[clean_df.Decade >= 1970]
# What was the most expensive film made prior to 1970
# print(old_films_df.sort_values('USD_Production_Budget', ascending=False))


"""Relationship between the movie budget and the worldwide revenue 
using linear regression"""
# Old Movies
# plt.figure(figsize=(6, 4), dpi=200)
# with sns.axes_style('whitegrid'):
#     sns.regplot(data=old_films_df,
#                 x='USD_Production_Budget',
#                 y='USD_Worldwide_Gross',
#                 scatter_kws={'alpha': 0.4},
#                 line_kws={'color': 'red'}
#                 )
#     plt.show()

# New Movies
# plt.figure(figsize=(6, 4), dpi=200)
# with sns.axes_style('darkgrid'):
#     ax = sns.regplot(data=new_flims_df,
#                      x='USD_Production_Budget',
#                      y='USD_Worldwide_Gross',
#                      color='#2f4b7c',
#                      scatter_kws={'alpha': 0.3},
#                      line_kws={'color': '#ff7c43'})
# 
#     ax.set(ylim=(0, 3000000000),
#            xlim=(0, 450000000),
#            ylabel='Revenue in $ billions',
#            xlabel='Budget in $100 millions')
#     plt.show()

regression = LinearRegression()
# Explanatory Variable(s) or Feature(s)
X = pd.DataFrame(new_flims_df, columns=['USD_Production_Budget'])

# Response Variable or Target
y = pd.DataFrame(new_flims_df, columns=['USD_Worldwide_Gross'])

# regression.fit(X, y)
# print(regression.intercept_)
# print(regression.coef_)
# print(regression.score(X, y))

X1 = pd.DataFrame(old_films_df, columns=['USD_Production_Budget'])
y1 = pd.DataFrame(old_films_df, columns=['USD_Worldwide_Gross'])
regression.fit(X1, y1)
print(regression.intercept_)
print(regression.coef_)
print(regression.score(X1, y1))

budget = 350000000
revenue_estimate = regression.intercept_[0] + regression.coef_[0, 0]*budget
revenue_estimate = round(revenue_estimate, -6)
print(revenue_estimate)