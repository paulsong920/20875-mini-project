import pandas

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['total'].replace(',','', regex=True))
# print(dataset_1) #This line will print out your data


"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""
high_temp = dataset_1.iloc[:,2]
low_temp = dataset_1.iloc[:,3]
precipitation = dataset_1.iloc[:,4]
brook =  dataset_1.iloc[:,5]
manh = dataset_1.iloc[:6]
willi = dataset_1.iloc[:,7]
queen = dataset_1.iloc[:,8]
total = dataset_1.iloc[:,9]
# Created a separate list for all collumns starting from temperatures
mon = 0
tue = 0
wed = 0
thu = 0
fri = 0
sat = 0
sun = 0

hist_days = []
for x in range(len(total)):
    if x % 7 == 0:
        fri = fri + int(total[x])
    elif x % 7 == 1:
        sat = sat + int(total[x])
    elif x % 7 == 2:
        sun = sun + int(total[x])
    elif x % 7 == 3:
        mon = mon + int(total[x])
    elif x % 7 == 4:
        tue = tue + int(total[x])
    elif x % 7 == 5:
        wed = wed + int(total[x])
    elif x % 7 == 6:
        thu = thu + int(total[x])

print(mon)



    
