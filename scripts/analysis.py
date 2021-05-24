# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

def expand_list(df, list_column, new_column): 
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = (
      [idx for idx, col in enumerate(df.columns)
       if col != list_column]
    )
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = (
      [item for items in df[list_column] for item in items]
      )
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df

def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

# preliminary file reading

path = "/Users/sumant/Python/steam/allpages.json" # or any host computer path
file = open(path,'r')
js = file.read()

# creation of aggregate data frame and summary plots

# adding year variable

data_list = json.loads(js)
df = pd.DataFrame(data_list)
df['time'] = pd.to_datetime(df['time'])
df['year'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y')
df['year'] = df['year'].astype(float)
year = df["year"]

plt.figure(1)
plt.style.use('fivethirtyeight')
plt.hist(year, edgecolor='black')
plt.title('Blog Postings per Year')
plt.ylabel('Frequency')
plt.savefig('blogposts.svg')

# adding comment variable

#year.plot(kind="hist", edgecolor='black')
df['comments'] = df['comments'].replace({' Comments': ''}, regex=True)
df['comments'] = pd.to_numeric(df['comments'],errors='coerce')
comments = df["comments"]

plt.figure(2)
plt.style.use('fivethirtyeight')
plt.hist(comments, edgecolor="black", bins=20)
plt.title('Distribution of Comments per Blog Posting')
plt.ylabel('Frequency')
plt.savefig('distribution.svg')

# this information is not plot worthy. Most authors have not contributed as much as
# the main two authors who themselves have similar engagement to their articles
author_comparison = df.groupby('author')['comments'].mean() 
author_comparison.to_csv('author_comparison.csv')

# comments by year

year_comparison = df.groupby('year')['comments'].sum()
year_comparison.to_frame()
year_comparison = year_comparison.rename_axis('year').reset_index()
plt.figure(3)
plt.style.use('fivethirtyeight')
fig = plt.figure(3)
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(year_comparison["year"],year_comparison["comments"])
plt.title('Comments per Year')
plt.xlabel('Year')
plt.ylabel('Comments')
plt.savefig('engagement.svg')

# main for loop

year = 2004
output = pd.DataFrame()

for i in range(1,19):
    filtered = df[df["year"] == year]
    df1 = to_1D(filtered['tags']).value_counts()
    df1 = df1.to_frame()
    df1 = df1.rename(columns = {0:f'tagfreq{year}'})
    df1 = df1.rename_axis('tags').reset_index()
    expanded_df = expand_list(filtered,"tags","expanded")
    df2 = expanded_df.groupby('expanded')['comments'].sum()
    df2 = df2.to_frame()
    df2 = df2.sort_values(by="comments", ascending = False)
    df2 = df2.rename(columns = {"comments":f'tagsum{year}'})
    df2 = df2.rename_axis('tags').reset_index()
    df1df2 = pd.concat([df1,df2], ignore_index=False, axis=1)
    output = pd.concat([output,df1df2], ignore_index=False, axis=1)
    year = year + 1

output.to_csv('output.csv')

# creation of aggregate data comparison

df1 = to_1D(df['tags']).value_counts()
df1 = df1.to_frame()
df1 = df1.rename(columns = {0:'tagfreq'})
df1 = df1.rename_axis('tags').reset_index()
expanded_df = expand_list(df,"tags","expanded")
df2 = expanded_df.groupby('expanded')['comments'].sum()
df2 = df2.to_frame()
df2 = df2.sort_values(by="comments", ascending = False)
df2 = df2.rename(columns = {"comments": "tagsum"})
df2 = df2.rename_axis('tags').reset_index()
df1df2 = pd.concat([df1,df2], ignore_index=False, axis=1)
df1df2.to_csv('aggregate.csv')



    
