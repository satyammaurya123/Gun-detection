import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset =pd.read_csv(r"C:\Algo\Copy of sonar data.csv", header = None)

# print(dataset.head(10))
# print(dataset.shape)
#print(dataset.describe()) # describe is use for statistical measures of the data
# print(dataset[60].value_counts())
#print(dataset.groupby(60).mean())

# Spearting the data and labels 

x = dataset.drop(columns=60,axis=1) # axis is used specifies that we are dropping a column instead of a row
y = dataset[60]
# print(x)
# print(y)

# Training and testing the data 
X_train, X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.1, stratify=y,random_state=1)

# print(x.shape,X_train.shape,X_test.shape)

sonar_testing = LogisticRegression()
sonar_testing.fit(X_train,Y_train)

# Accuary data on training data 
X_trian_predicition = sonar_testing.predict(X_train)
training_data_accuracy = accuracy_score(X_trian_predicition,Y_train)
# print("training data accuracy is: ",training_data_accuracy*100)

#Accuracy data on testing data

X_test_predicition = sonar_testing.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_predicition,Y_test)
# print("training data accuracy is: ",testing_data_accuracy*100)

#Making a Predicating system 
input_data = (0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.0080,0.0107,0.0161,0.0133) 

#  change the input_data to numpy array 
input_array_numpy = np.asarray(input_data)

# reshape the np array as we predicting for one instance 
input_data_reshaped = input_array_numpy.reshape(1,-1)

prediciton = sonar_testing.predict(input_data_reshaped)
print(prediciton)

if (prediciton[0]== 'R'):
    print("This object is Rock")
else:
    print("This object is Mine")

