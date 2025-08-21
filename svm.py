import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

# person, pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age, outcome
df = pd.read_csv('diabetes.csv')

#replace zeros with median
replaceZeros = ["Glucose","BloodPressure","SkinThickness"]

for column in replaceZeros:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].median(skipna = True))
    df[column] = df[column].replace(np.NaN, mean)

#use pearson median skew = 3 * ((mean - median)/ std)
#normal dist: z score (|zscore| >= 3 is considered outlier)
#skewed: IQR (Data points that fall below Q1−1.5×IQR or above Q3+1.5×IQR are considered outliers.)
for column in df.columns:
    mean = df[column].mean()
    median = df[column].median()
    std = df[column].std()
    pearson = 3 * ((mean - median)/ std)
    print(column, pearson, 'is skewed: ', (abs(pearson) >= 0.4))
    if(abs(pearson) >= 0.4):
        #skewed so use IQR
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[column] = df[column].clip(lower, upper) #replace with two bounds
    else:
        #normal so zscore
        lower = mean - 3 * std
        upper = mean + 3 * std
        df[column] = df[column].clip(lower, upper)


#load into array
arr = df.to_numpy()
# print(arr.shape)

#split outcome and data
labels = arr[:, 8]
data = arr[:, 0:8]


# print(labels.shape)
# print(data)

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.15, random_state=0)
print(data_train.shape)
print(labels_train.shape)
print(data_test.shape)
print(labels_test.shape)

clf = svm.SVC(kernel='rbf')
clf.fit(data_train, labels_train)

count = 0
n = data_test.shape[0]
print(n)
for i in range(0, n):
    res = clf.predict(data_test[i,:].reshape(1,-1))[0]
    if(res == labels_test[i]):
        count += 1
print("accuracy: ", (count/n))

    