 #1...data preprocessing
#classification tamplate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')

#creat fature matrix
X =dataset.iloc[:,3:13].values #featurs
y =dataset.iloc[:,13].values   #vectors

#encoded categorical variables
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]
#splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size=0.2 ,random_state=0)

#fature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#2... Make the ANN

#import the keras library ana package
import keras
from keras.models import Sequential  #initialize
from keras.layers import Dense       #creat layers

#initializing ANN
classifier=Sequential()

#Adding the input layer & first hidden layer
classifier.add(Dense(output_dim=6 ,init ='uniform', activation='relu', input_dim=11))         #(no of i/p + no of o/p)/2

#Adding the second hidden layer
classifier.add(Dense(output_dim=6 ,init ='uniform', activation='relu' ))

#Adding the output layer
classifier.add(Dense(output_dim=1 ,init ='uniform', activation='sigmoid' ))

#compiling ann
classifier.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'] )


#fitting ANN to the training set
classifier.fit(X_train ,y_train ,batch_size=10 , nb_epoch=100)

#3...make the prediction and evaluatin the model

#predicting the test set
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test , y_pred )
