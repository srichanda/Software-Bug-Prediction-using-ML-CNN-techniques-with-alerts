
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('Dataset/1 spring-framework/2015-6.csv')

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

#plotPerColumnDistribution(dataset, 40, 5)
'''
print(type(dataset.isnull().sum()))
fig, ax = plt.subplots()
sns.lineplot(data=dataset.isnull().sum())
fig.autofmt_xdate() 
plt.show()
'''
dataset.fillna(0, inplace = True)
cols = ['QualifiedName','Name','Complexity','Coupling','Size','Lack of Cohesion']
le = LabelEncoder()
dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))
dataset[cols[3]] = pd.Series(le.fit_transform(dataset[cols[3]].astype(str)))
dataset[cols[4]] = pd.Series(le.fit_transform(dataset[cols[4]].astype(str)))
dataset[cols[5]] = pd.Series(le.fit_transform(dataset[cols[5]].astype(str)))
Y = dataset.values[:,2]
dataset.drop(['Complexity'], axis = 1,inplace=True)
X = dataset.values







