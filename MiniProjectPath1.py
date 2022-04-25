import pandas
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''

def regression(data, prediction):

    avg = statistics.mean(data)
    std = statistics.stdev(data)
    Xtrain = data[0:172]
    Xtest = data[172:]
    
    Xtrain = (np.array(Xtrain)).reshape(172,1)
    Xtest = (np.array(Xtest)).reshape(42,1)
    Xtrain = (Xtrain - avg) / std
    Xtest = (Xtest - avg) / std

    avg = statistics.mean(prediction)
    std = statistics.stdev(prediction)

    Ytrain = prediction[0:172]
    Ytrain = (np.array(Ytrain)).reshape(172,1)
    Ytrain = (Ytrain - avg) / std
    ytest = prediction[172:]

    MSE = []
    r2 = []
    model = []
    lmbda = np.logspace(-1, 3, num = 51)
    for val in lmbda:

        reg = Ridge(alpha = val ,fit_intercept=True)
        reg.fit(Xtrain,Ytrain)  #Creating model using training data

        ypredictionTest = reg.predict(Xtest) # testing model on testing data
        ypredictionTest = ypredictionTest * std + avg
        mse = mean_squared_error(ytest, ypredictionTest) #  Mean Squared error
        rSquared = r2_score(ytest, ypredictionTest)
        model.append(ypredictionTest)
        MSE.append(mse)
        r2.append(rSquared)
    
    ind = np.argmin(MSE)
    mse = MSE[ind]
    rSquared = r2[ind]
    
    return mse, rSquared

def stringtoint(lists):
    x = [int(a) for a in lists]
    return x

dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))

# print(dataset_1) #This line will print out your data


"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""
df = pandas.DataFrame(dataset_1)
# Creating a list of integers for total
high_temp = stringtoint(df['High Temp'].tolist())
low_temp = stringtoint(df['Low Temp'].tolist())
precipitation = stringtoint(df['Precipitation'].tolist())
brook = stringtoint(df['Brooklyn Bridge'].tolist())
manh = stringtoint(df['Manhattan Bridge'].tolist())
willi = stringtoint(df['Williamsburg Bridge'].tolist())
queen = stringtoint(df['Queensboro Bridge'].tolist())
total = df['Total'].tolist()
totalTraffic = stringtoint(total)


# Creating a column for if it rained
rain = []
for y in precipitation:
    if y > 0:
        rain.append(1)
    else:
        rain.append(0)
df['rain'] = rain

# Creating a column for average temperature of the day
avgtemp = []
for x in range(len(high_temp)):
    temp = (high_temp[x] + low_temp[x]) / 2
    avgtemp.append(temp)
df['Average Temp'] = avgtemp

# Creating the descriptive statistic of all collumns
print(df.describe())


days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Find the sum of totals by day and plot as histogram
dataset_1[['Day', 'Total']].groupby('Day').sum().reindex(days_of_week).plot(kind='bar', legend=None, title='Amount of Bicyclists per Day', xlabel='Day', ylabel='Bicyclists')
plt.show()
# Print the Dataset being plotted
print(dataset_1[['Day', 'Total']].groupby('Day').sum().reindex(days_of_week))

#####################################################
#Question 2
temp_prediction = regression(avgtemp, totalTraffic)
tempMSE = temp_prediction[0]
tempr2 = temp_prediction[1]

high_prediction = regression(high_temp, totalTraffic)
highMSE = high_prediction[0]
highr2 = high_prediction[1]

low_prediction = regression(low_temp,totalTraffic)
lowMSE = low_prediction[0]
lowr2 = low_prediction[1]

rain_prediction = regression(precipitation, totalTraffic)
rainMSE = rain_prediction[0]
rainr2 = rain_prediction[1]

print("Average Temperature MSE : ", tempMSE)
print("Average Temperature R^2 : ", tempr2)
print("High Temperature MSE : ",highMSE)
print("High Temperature r^2 : ", highr2)
print("Low temperature MSE : ", lowMSE)
print("low Temperature r^2 : ", lowr2)
print("Precipitation MSE : ", rainMSE)
print("Precipitation r^2 : ", rainr2)




