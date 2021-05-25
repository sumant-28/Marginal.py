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

# creation of final plot

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

def addconnection(i,j,c):
  return [((-1,1),(i-1,j-1),c)]

def drawnodes(s,i):
  global ax
  if(i==1):
    color='r'
    posx=1
  else:
    color='b'
    posx=-1

  posy=0
  for n in s:
    plt.gca().add_patch( plt.Circle((posx,posy),radius=0.05,fc=color))
    if posx==1:
      ax.annotate(n,xy=(posx,posy+0.1))
    else:
      ax.annotate(n,xy=(posx-len(n)*0.1,posy+0.1))
    posy+=1

ax=plt.figure().add_subplot(111)
plt.title('Bipartite Graph of Post Frequency vs Accumulated Comments')
set1=['Travels','Weblog','Television','Games','Music','Sports','Film','The Arts',
      'Religion','Travel','Food and Drink','Web/Tech','Science','Philosophy','Books',
      'Medicine','Data Source','Education','History','Political Science','Law','Current Affairs',
      'Economics']
set2=['W', 'Tr','Te','G','Sp','M','F','R','Tr',
      'TA','F&D','P','M','Sc','DS','W/T',
      'H','Ed','PS','L','B',
      'CA','Ec']
plt.axis([-2,2,-1,max(len(set1),len(set2))+1])
frame=plt.gca()
frame.axes.get_xaxis().set_ticks([])
frame.axes.get_yaxis().set_ticks([])

drawnodes(set1,1)
drawnodes(set2,2)

connections=[]
connections+=addconnection(23,23,'g')
connections+=addconnection(22,22,'g')
connections+=addconnection(21,15,'g')
connections+=addconnection(20,21,'g')
connections+=addconnection(19,20,'g')
connections+=addconnection(18,18,'g')
connections+=addconnection(17,19,'g')
connections+=addconnection(16,12,'g')
connections+=addconnection(15,17,'g')
connections+=addconnection(14,13,'g')
connections+=addconnection(13,16,'g')
connections+=addconnection(12,14,'g')
connections+=addconnection(11,11,'g')
connections+=addconnection(10,8,'g')
connections+=addconnection(9,10,'g')
connections+=addconnection(8,9,'g')
connections+=addconnection(7,7,'g')
connections+=addconnection(6,5,'g')
connections+=addconnection(5,6,'g')
connections+=addconnection(4,4,'g')
connections+=addconnection(3,3,'g')
connections+=addconnection(2,1,'g')
connections+=addconnection(1,2,'g')

for c in connections:
  plt.plot(c[0],c[1],c[2])

plt.savefig('bipartite.svg')

    
