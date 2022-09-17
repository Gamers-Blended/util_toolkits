# data handling libraries
import pandas as pd
import numpy as np
import glob, os

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure # adjust size
import plotly.express as px
from mpl_toolkits import mplot3d # 3d plot
import geopandas as gpd # geodata

# time series
import datetime

# import stats libraries
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest


### Settings ###
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# warnings
# disable chained assignments
pd.options.mode.chained_assignment = None # else need to use df.copy()


### WORKING DIRECTORY ###
# get path of current working directory
import os
path = os.getcwd()
  
# get list of all files and directories in current working directory
dir_list = os.listdir(path)

# change working directory
os.chdir(path)
print("Current working dir : %s" % os.getcwd())


#####################
### DATA CLEANING ###
#####################

# if encoding needed
import chardet
# Use chardet to identify encoding
rawdata = open("dataset.csv", 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
charenc


# read CSV into DataFrame
df = pd.read_csv("../path/data.csv")
df = pd.read_csv('data.csv', index_col=0) # 1st column is index

# read Excel into DataFrame
# multiple sheets
xls = pd.ExcelFile('../path/data.xlsx')
df1 = pd.read_excel(xls, 'sheet_1_name')
df2 = pd.read_excel(xls, 'sheet_2_name')
df = df1.append(df2, ignore_index=True)

# read txt files
with open("txt_file_name.txt", 'r') as f:
    for line in f:
        # look at line in loop
        print(line, end='')


# get string of file names that ends with .txt
filename_lst = []

os.chdir("folder_containing_txt_files/")
for file in glob.glob("*.txt"):
    filename_lst.append(file)
filename_lst

# get a list of strings from txt files
all_txt_lst = []

for text in filename_lst:
    with open(text, 'r', encoding='utf-8') as f:
        curr_text = f.read()
        all_txt_lst.append(curr_text.strip())
all_txt_lst


# replace values in DataFrame with .replace
df = df.replace("No", 0)
df = df.replace("Yes", 1)

df = df.replace("Positive", 1)
df = df.replace("Negative", 0)

# consider Gender column as isMale
df = df.replace("Male", 1)
df = df.replace("Female", 0)

# replace ?/! with .
df['column_name'] = df['column_name'].str.replace('!','.')
df['column_name'] = df['column_name'].str.replace('?','.')

# rename columns
replace = {"Gender": "ismale"}
df.rename(columns=replace, inplace=True)


### MISSING DATA ###
# check for missing values
# to get the number of missing values in each column:
df.isnull().sum()


# remove rows with missing data
df = df.dropna()
# remove missing data from specific columns
df = df[df['column_name'].notna()]
# remove all unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# replace missing values with 0
df['column_name'] = df['column_name'].fillna(0)

# IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# declare a variable containing an IterativeImputer object
imp_mean = IterativeImputer(random_state=0)

# fit DataFrame with numerical features into IterativeImputer object
imp_mean.fit(num_df)

# transform DataFrame with numerical features and assign it to a new variable
num_df_imputed_array = imp_mean.transform(num_df)

# transform array into a DataFrame
num_df_imputed = pd.DataFrame(data=num_df_imputed_array,    # values
                              columns=num_df.columns)  # 1st row as the column names
####


# check dtypes of columns
df.dtypes
df.info()
df.index.inferred_type # dtype of index column


# change column(s) dtype to string 
df['column_name'] = df['column_name'].astype(str)
# or
change_columns = ['column_name1', 'column_name2'] # creates list of all column headers
df[change_columns] = df[change_columns].astype(str)


# lowercase everything in columns
df.columns = df.columns.str.lower()
# lowercase everything in lists
items_lst = list(map(lambda x: x.lower(), lst_to_convert))


# convert string representation of list into acutal list
import ast
df['column_name'] = df['column_name'].apply(lambda row: ast.literal_eval(row))


# remove column(s)
df.drop('column_name', axis=1, inplace=True)
df.drop(['column_name1', 'column_name2'], axis=1, inplace=True)

# remove column(s) containing certain substrings in column name
df.iloc[:, ~df.columns.str.contains('unwanted_substring')]


# drop duplicates
df.drop_duplicates(subset='column_name', keep="last")
# or
df.drop_duplicates(subset=['column_name1', 'column_name2'], keep='first')


# set index
df.set_index('index_column_name', inplace=True)


# create index column
df.insert(loc=0, column='index', value=np.arange(len(df)))


# get substrings
# eg: "a2b"
a = string_text.split('2')[0] # idx 1 for right substring of '2': b


# remove prefix
df.columns = df.columns.str.lstrip("string_text")
df['string_text'] = df['string_text'].map(lambda x: x.lstrip('+-'))
# remove suffix
df.columns = df.columns.str.rstrip("string_text")
df['string_text'] = df['string_text'].map(lambda x: x.rstrip('aAbBcC'))


# remove special characters
value = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", value)
# or
df['column_name'] = df['column_name'].str.replace(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", regex=True)


# get only integers from string
int(''.join(filter(str.isdigit, string_text)))
# or
int(filter(str.isdigit, string_text))


# get only numerical features
df_numerics = df.select_dtypes('number')
# object, datetime, timedelta, category, bool, float64, int64


# get column names with same prefix
cols_with_same_prefix = [col for col in df if col.startswith('prefix_string')]


# filter DataFrame to contain only values from a list
data = df[df['column_name'].isin(target_list)]


# split string to list
df['column_name'].apply(lambda x: x.split('.'))


# value_counts to DataFrame (df)
df = pd.DataFrame(value_count_series, columns=['column_name'])

# dictionary to DataFrame (df)
df = pd.DataFrame(dictionary.items(), columns=['column_name1', 'column_name2'])
df = df.set_index('index_column_name')
# or
df = pd.DataFrame.from_dict(dictionary, orient="columns")

# numpy array to DataFrame (df)
df = pd.DataFrame(numpy_array, columns=['column_name1', 'column_name2'])

# single list to DataFrame (df)
df = pd.DataFrame(lst1, columns=['column_name1'])
# multiple lists to DataFrame
df = pd.DataFrame(list(zip(lst1, lst2)),
               columns=['column_name1', 'column_name2'])


# merge multiple DataFrames (df)
# standard 2 dfs
merged = pd.merge(left_df, right_df, left_on='left', right_on='right')

# compile list of DataFrames to merge
data_frames = [df1, df2, df3]
df_merged = pd.concat(data_frames, axis='columns') # or axis=1, axis=0 is row-wise

# merge 2 DataFrames together by index
final_df = pd.merge(df1, df2, left_index=True, right_index=True)

# concantenate row-wise
df_new_row = pd.DataFrame(data=[['new_val1','new_val2']],
                     columns=['column_name1', 'column_name2'])
final_df = pd.concat([df, df_new_row],ignore_index=True) # reset index
# or (another way to reset index)
final_df.reset_index(drop=True, inplace=True)

# concantenate string columns
str_columns = ['column_name1', 'column_name2']
df_merged['combined'] = df_merged[str_columns].apply(''.join, axis=1)


# concantenate numpy arrays
single_numpy_array = np.concatenate(lst_of_numpy_arrays, axis=0)


# turn string list into list (read_csv turns lists into strings, need to convert back)
df['list_column'] = df['list_column'].apply(lambda row: ast.literal_eval(row))


# apply functions to Series
df['column_name'].apply(lambda x: function_name(x))
df['column_name'].apply(function_name)
df['column_name'].map(function_name)
# apply functions to DataFrame
df[['column_name1', 'column_name2']].apply(function_name)
df[['column_name1', 'column_name2']].apply(lambda x: x*2)
df[['column_name1', 'column_name2']].applymap(function_name)


# zip multiple lists
combined_lst = ["{}{}{}".format(a, b, c) for a, b, c in zip(lst1, lst2, lst3)]
combined_lst


# combine multiple lists
joined_list = lst_1 + lst_2


# loop through DataFrame
for index, row in df.iterrows():
    print(row['column_name'])


# flatten a list of lists (single character) eg: [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
flat_list = [x for xs in target_lst for x in xs]
# or 
flat_list = np.concatenate(target_lst).ravel()

# flatten a list of lists containing strings
def flatten(target_lst):
    rt = []
    for i in target_lst:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt


# get index with highest value of list
highest_value = np.argmax(list_of_items, axis=1)


# Time Series #
###############
# DateTimeIndex, index contains DateTime

# read DataFrame and parse dates
df = pd.read_csv("data.csv", index_col="index_column_name", parse_dates=True)

# check datatime dtype
df.index.inferred_type == "datetime64"


# convert datetime to string
date.strftime("%Y-%m-%d") # YYYY-MM-DD
date.strftime("%H") # 00:00:00
date.strftime("%M")


# create DateTime object for 2012-01-01
today = datetime.datetime(2012, 1, 1)

# generate a date range in 5-min intervals
pd.date_range(start='2019-01-01', end='2019-01-31', freq='5T')

# convert index column to datetime
df.index = pd.to_datetime(df.index)


# calculate difference in terms of days
df['days_difference'] = df['day1_col'] - df['day2_col']
df['days_difference'] = df['days_difference'].dt.days


# month
df['month'] = df.index.dt.month
# day of week
df['dayofweek'] = df.index.dt.dayofweek
# day of month
df['day_of_month'] = df.index.dt.day
# hour
df['hour'] = df.index.dt.hour
# minute
df['minute'] = df.index.dt.minute


# time series resampling
# resample by day, pick mean/median
resampled_mean = df.resample('D').mean()
resampled_median = df.resample('D').median()

# resample by month, containing sum of quantities
df = df.resample('M').sum()


# export DataFrame to CSV
df.to_csv("/path/file_name.csv", index=None)


###########
### EDA ###
###########
# example questions
# What was the most popular item, i.e. bought by most number of unique customers?
# What was the item that is bought in the largest quantity?


# get value by index, column name
df.loc[index_number, 'column_name']
df.iloc[index_number, column_number] # only numerical values

# get number of rows of DataFrame that contain ...
(df['column_name'] == 'target_val').sum()


# filter DataFrame
df[df['column_name'] > 1]
df[(df['column_name'] > lower_bound) & (df['column_name'] < upper_bound)]

# results contain substring
df[df['column_name'].str.contains('wanted_substring')]


# groupby x, sum
df_grouped = df.groupby(by='x').count().reset_index() # sum()
# sort by y
df.sort_values(by='y', axis=0, ascending=False, inplace=True)


# groupby aggregate functions
df_grouped = df.groupby('x').agg({'column_name': 'min'}) # sum, size

# multiple aggregate functions
def square_root(p): # example function
    return np.mean(np.sqrt(p))

df.groupby('x')['col_to_apply_function'].agg(['count', 'mean', square_root])


# pivot
df.pivot(index='column_name1', columns='column_name2', values='value')


# summary stats (count, mean, std, min, max, 25-50-75)
df.describe()
# summary includes catergorical data
df.describe(include='all')


# get a list of unique items from column
df['column_name'].unique()
# or
set(df['column_name'])
# get number of unique items from column
df['column_name'].nunique()

# get number of unique items from numpy array
np.unique(numpy_array)


# get count - value_counts
df['column_name'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
# get count in terms of percentage
df['column_name'].value_counts(normalize=True)
# get count - group_by
count = df.groupby('x').size()
count.sort_values(axis=0, ascending=False)


# use dictionary to count items in a list
from collections import Counter
Counter(lst)
# get value of the key 'a' in Counter
Counter(lst)['a']


# get quintile range
# another way of representing datapoints is by binning them,
# i.e. turning continuous values into categorical values
## standard
pd.qcut(df['column_name'], q=5)

## manually define quartile
lst_of_labels = np.arange(5, 0, -1) # backwards because lower value, higher score
df['column_name_bin'] = pd.qcut(df['column_name'], 5, labels=lst_of_labels)

## if lots of repeated values, use rank
lst_of_labels = np.arange(1, 6)
df['column_name_rank'] = pd.qcut(df['column_name'].rank(method='first'), q=5, labels=lst_of_labels)


# statistical test of difference
# distribution for two series are not normal, use Mann-Whitney U test
# if normal distribution, use t-test
# ff p-value is 0.05 and below, reject null hypothesis that two distributions are the same
from scipy.stats import mannwhitneyu

mannwhitneyu(series_1, series_2)
# if p-value < 0.05, there is statistical significance in the difference


##########################
### DATA VISUALISATION ###
##########################
# to export png file
# add this at end of code snippet
plt.savefig("name_of_plot.png")


# look at distribution
# examine numerical values with histogram - check if Normal Distribution
# add second plt.hist() for second series
plt.hist(df["column_name"], range=[0, 200], bins=10)
plt.ylabel('Count')
plt.xlabel('column_name')
plt.show()


df["column_name"].mean()
df["column_name"].median()
 
# 2 series in same histogram
series_1 = df['column_name1']
series_2 = df['column_name2']

plt.hist(series_1, alpha=0.5, label='series_1')
plt.hist(series_2, alpha=0.5, label='series_2')

plt.ylabel('Count')
plt.xlabel('series_1')
plt.legend(loc='upper right')
plt.show()


# dataframe built-in plot
df["column_name"].plot()


### seaborn
# displot
sns.displot()
# line plot
sns.lineplot(data=df, x='column_name1', y='column_name2')


# line graph
# add second plt.plot() for second series
plt.figure(figsize=(20,10))
plt.plot(df.index, df["y_column"], label='label')
# plt.semilogy(np.log10(x), np.log10(y), label='label')
# plt.loglog(x, y, label='label')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
# plt.yscale('log')
# plt.xscale('log')
plt.show()


# log-transform
# if skewed distribution, transform to Normal Distribution
# if using linear regression, ensure independent variables are normally distributed
# plot log transformed version
df['x_log'] = np.log(df['x'])
plt.hist(df["x_log"], range=[0, 10], bins=10)
plt.ylabel('Count')
plt.xlabel('Log_x')
plt.show()


### univariate analysis (single variable)
# create countplot
# look at proportions
ax = sns.countplot(x = df["x"])
ax.set(xlabel='x', ylabel='Count')
plt.title('variable')
plt.xticks(rotation=90)
# sns.despine() # remove top and right spines from plot
plt.show()

# iteratively plot countplot
for column in columns:
    sns.countplot(x = df[column])
    plt.title(column)
    # sns.despine() # remove top and right spines from plot
    plt.show()


### bivariate analysis
# Plot 'x' against 'y' with a scatter plot
plt.figure(figsize=(20,10))
df.plot(x='x', y='y', kind='scatter')
# or
# plt.scatter(x, y)
# for i, txt in enumerate(text_lst):
#     plt.annotate(txt, (x[i], y[i]))
# plt.title("Predictions - Linear Reg")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.show()


# test of independence between 2 categorical features
# is x related to y (both categorical)?
# is there a relationship?
x_y_crosstab = pd.crosstab(df['target'], df['x'])
x_y_crosstab

# chi sq contingency test
chi2_contingency(x_y_crosstab)
# tuple with 4 values
# #1 chi sq stats,
# #2 p-value, p-value = 0.12 -> cannot say there is a relationship between x and y.
# #3 degree of freedom,
# #4 expected array if everything is independent, any deviation from this will be deemed dependent

# Since p-value is low and expected array deviates significantly from cross_tab, there is a relationship between x and y.
# Therefore likely to be NOT independent from each other


# boxplot
## 1 series
sns.set(rc={'figure.figsize':(20,10)})
ax = sns.boxplot(x=df['x'], y=df['y'], showfliers=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("X_Axis", fontsize=20)
ax.set_ylabel("Y_Axis", fontsize=20)

## 2 series
sns.set(rc={'figure.figsize':(20,10)})
ax = sns.boxplot(x=df['x'], y=df['y'], hue=df['group'], showfliers=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("X_Axis", fontsize=20)
ax.set_ylabel("Y_Axis", fontsize=20)


# scatterplot
## 1 series
plt.figure(figsize=(20,10))

plt.scatter(df['x'], df['y'], label = "label")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

## 2 series
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(111)

ax1.scatter(df['x'], df['y'], c='b', marker="s", label = "group1")
ax1.scatter(df['x'], df['y'], c='r', marker="o", label = "group2")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()


# bar plot barh (horizontal)
plt.figure(figsize=(20,10))
plt.barh(y=df['variable_index'].value_counts().sort_values().index,
    width=df['value_counts'].value_counts().sort_values())


# relationship between cat and numerical
sns.boxplot(df['x'], df['y'])


# quick mean check
do_not_have_label = df[df['label']==0]
do_not_have_label['x'].mean()
do_not_have_label['x'].median()
# compare this against ...
have_label = df[df['label']==1]
have_label['x'].median()


# stat test depends on distribution
# qqplot allows you to assess whether qty is normal or not
qqplot(df['x'], fit=True, line='s')
plt.show()
# distribution is normal if most points lie within 45 degree line


# conduct z test of difference for Normally Distributed data
ztest(have_label['x'], do_not_have_label['x'])
# (z score, p-value for stats test)

# p-value < 0.05 -> reject H0 that there is no difference between x and label -> there is a difference between their means


# get correlation plot
df.corr()
# or
sns.heatmap(df.corr())


# 3d plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')

ax.scatter(df1['x_property1'],
           df1['y_property1'],
           df1['z_property1'],
          c=lst_of_labels) # coloured scatter points
ax.scatter(df2['x_property2'],
           df2['y_property2'],
           df2['z_property2'],
           c='Orange', s=22**2) # orange markers with different size

plt.title("3-D Plot")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


# smoothing curves by interpolation
from scipy.interpolate import make_interp_spline, BSpline

## for single curve
x = df['index']
y = df["y"]
# last parameter represents number of points to make between x.min and x.max
# linspace(start, stop, num) returns array with num evenly-spaced numbers spanning from start to stop, inclusive
x_new = np.linspace(x.min(), x.max(), 20) 

spl = make_interp_spline(x, y, k=3)  # type: BSpline
power_smooth = spl(x_new)

plt.plot(x_new, power_smooth)
plt.show()

## for multiple curves
plt.figure(figsize=(25,9))

label_lst = ['label1', 'label2', 'label3', 'label4', 'label5']
label_idx = 0

for df in df_lst: # df_lst is a list of DataFrames
    curr_idx_col = df['index']
    curr_val_col = df['column_name']

    x_new = np.linspace(0.0, 1.0, num=20)
    spl = make_interp_spline(curr_idx_col, curr_val_col, k=3)
    power_smooth = spl(x_new)
    plt.plot(x_new, power_smooth, label=label_lst[label_idx])
    label_idx += 1

plt.legend(loc='upper right')
plt.show()


# fitting curve to polynomial
import numpy.polynomial.polynomial as poly

x = df['index']
y = df["y"]
x_new = np.linspace(x[0], x.values[-1], num=len(x)*10)

coefs = poly.polyfit(x, y, 5)
ffit = poly.polyval(x_new, coefs)
plt.plot(x_new, ffit)


# word cloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# declare a variable containing STOPWORDS from wordcloud library
stopwords = set(STOPWORDS)
stopwords.update(["2", "&", "set", "4", "10"])


# create and display wordcloud
text = items_str

# generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(text)

# display generated image
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# visualise image files
%pylab inline
import matplotlib.image as mpimg

img = mpimg.imread('path/to/image.png')
imgplot = plt.imshow(img)
plt.show()

# or (for JSON containing URLs)
from PIL import Image
from io import BytesIO

img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.show()

# or (for URLs)
import urllib2

# create a file-like object from the url
f = urllib2.urlopen("http://matplotlib.sourceforge.net/_static/logo2.png")

# read the image file in a numpy array
a = plt.imread(f)
plt.imshow(a)
plt.show()


#######################
# FEATURE ENGINEERING #
#######################
# get number of characters in column
df['len_text'] = df['column_name'].apply(len)


# mean normalisation
normalized_df=(df-df.mean())/df.std()
normalized_df=(df-df.min())/(df.max()-df.min())


######################
### CLASSIFICATION ###
######################
# processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

# metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
# note: Healthcare, more impt to reduce false negative (illness not detected)


# normalisation
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df)
print(scaled_array)

df = pd.DataFrame(scaled_array, columns=['column_name1','column_name2']) # if want to convert numpy array to df

# encode catergorical data
df['x_num'] = df['x'].map({'class1': 1, 'class2': 2, 'class3': 3})
# or
# use LabelEncoder
variable_le = LabelEncoder()
df['column_name'] = variable_le.fit_transform(df['column_name'])
# to see classes
variable_le.classes


# target and feature(s) split
X = df.drop('label', axis=1)
y = df['label']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# begin model training
# start with DummyClassifier to establish baseline
dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)


# access DummyClassifier model
confusion_matrix(y_test, dummy_pred)
# [TN, FP,
# FN, TP]

# classification report
print(classification_report(y_test, dummy_pred))


# F1 Score
# Harmonic mean of precision and recall
# Best value at 1, worst score at 0
# Relative contribution of precision and recall to F1 score are equal
# F1 = 2 * (precision * recall) / (precision + recall)
# In multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on average parameter
f1_score(y_test, y_pred, average=None)


# feature importance
forest_model.feature_importances_
# print together with columns
X.columns

# build dataframe
pd.DataFrame({'feature': X.columns,
             'importance': forest_model.feature_importances_}).sort_values('importance',
                                                                    ascending=False)


# plot accuracy and val plots
# history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=4)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# KMeans
# declare a variable containing KMeans object with 10 clusters in parameter
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans = kmeans.fit(df)
# predict labels for each of rows in DataFrame
df['Label'] = kmeans.labels_

# create a DataFrame containing cluster centers' coordinates
cluster_center_df = pd.DataFrame(kmeans.cluster_centers_, columns=['column_name1','column_name2'])


### Clustering with KMeans ###
##############################
# plot WCSS curve
# create an empty list
wcss = []

# loop through a range between 1 to 15
for cluster_num in range(1,16):
    # declare a KMeans classifier object, with current value in range as number of clusters
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    kmeans = kmeans.fit(df)
    kmeans_wcss = kmeans.inertia_
    # append .inertia_ of trained model
    wcss.append(kmeans_wcss)

# plot a line plot containing range of clusters (x-axis), against list of WCSS scores (y-axis)
fig = plt.figure(figsize=(12,9))
plt.plot(range(1,16), wcss)
plt.title("Elbow Point Graph")
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
plt.show()


##################
### REGRESSION ###
##################
# processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# models
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans


# metrics
from sklearn.metrics import mean_squared_error, accuracy_score

accuracy_score(y_true, y_pred) * 100


#############################
### HYPERPARAMETER TUNING ###
#############################
from sklearn.model_selection import GridSearchCV


# define parameter grid
param_grid = {
    'n_estimators': [5, 10, 25, 50, 100],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': [9, 11, 13, 15, 17]
}

# declare a GridSearchCV object
gboost_clf = GradientBoostingClassifier(random_state=42)
clf = GridSearchCV(gboost_clf, param_grid, scoring='precision', n_jobs=4, cv=5)
clf.fit(X_train, y_train)

# get best parameters
clf.best_params_


###################
### UPSAMPLING  ###
###################
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

# upsample train data
sm = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


####################
### RESTFUL API  ###
####################
import requests

# declare request URL
request_url = "https://api.data.gov.sg/v1/transport/traffic-images?date_time=2019-01-01T00%3A00%3A00"

# make API request and save results in a variable
response = requests.get(request_url)
print(response.status_code)

# look at JSON from response
json_result = response.json()
json_result

# get DataFrame from JSON
df = pd.json_normalize(json_result["check_if_need_to_retrieve_keys"])
df

# example of getting appropriate API format
# general practice, start with first row
# then entire DataFrame
# format first datetime in list to string - ['2019-01-01T00%3A00%3A00']

## first row
date_range = pd.date_range(start='2019-01-01',  end='2019-01-02', freq='T')

date_lst = []
hour_lst = []
min_lst = []

date_lst.append(date_range[0].strftime("%Y-%m-%d") + 'T')
hour_lst.append(date_range[0].strftime("%H") + '%3A')
min_lst.append(date_range[0].strftime("%M") + '%3A00')

api_range = ["{}{}{}".format(a, b, c) for a, b, c in zip(date_lst, hour_lst, min_lst)]


## entire DataFrame
# call API and get a list of JSON objects
date_lst = []
hour_lst = []
min_lst = []

for curr_date in date_range:
  date_lst.append(curr_date.strftime("%Y-%m-%d") + 'T')
  hour_lst.append(curr_date.strftime("%H") + '%3A')
  min_lst.append(curr_date.strftime("%M") + '%3A00')

api_range = ["{}{}{}".format(a, b, c) for a, b, c in zip(date_lst, hour_lst, min_lst)]


# retrieve info from entire DataFrame
start = time.time()

json_lst = []

base_url = "https://api.data.gov.sg/v1/transport/traffic-images?date_time="

for t in api_range:
    # combine the base_url and the current date in the for loop
    date_time_call = base_url + t
    # make a get request using the combined URL
    response = requests.get(date_time_call)
    # get the JSON from the response of the get request
    json_result = response.json()
    # turn the variable into a DataFrame
    # df = pd.json_normalize(json_result["items"][0]['cameras'])
    # append the dataframe into the empty list above
    json_lst.append(json_result)
    
    print("done:" + date_time_call)

end = time.time()
print(end - start)
print(json_lst)


# combine list of JSON into a DataFrame
df_lst = []

for json_item in json_lst:
  curr_df = pd.json_normalize(json_item["check_if_need_to_retrieve_keys"])
  df_lst.append(curr_df)

df = pd.concat(df_lst)
df


### TO HIDE API KEYS ###
########################
# use dotenv library
# create a .env file (no name) with notepad
import os
from dotenv import load_dotenv

load_dotenv()

# eg: .env file has SECRET_KEY variable
# .env file is in working directory
SECRET_KEY = os.getenv('SECRET_KEY')


###################
### SAVE IMAGES ###
###################
# loop through URLs in dataframe
# uses URLs to make GET requests
def getImages(index, row, destination_path):
  row_num = index
  temp_url = row['image']
  temp_res = requests.get(temp_url)

  try:
    curr_img = Image.open(BytesIO(temp_res.content))
    curr_img.save(destination_path + str(row_num) + '.png')

  except:
    pass

