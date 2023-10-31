import pandas as pd
import numpy as np

#importing train dataset csv file
train_df = pd.read_csv('train.csv').drop(columns = ['Name','Ticket','Cabin','SibSp','PassengerId','Parch','Pclass','Embarked'])

train_df.dropna(inplace=True)
encoded_df = pd.get_dummies(train_df, columns = ['Sex']) # encoding: train_df -> encoded_df



#Random Forest Classifier Model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = encoded_df.drop(columns=['Survived']) #Separating the target variable.
y = encoded_df['Survived']

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 42)

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)

rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)



#importing test dataset csv file
test_df = pd.read_csv('test.csv')

test = test_df.drop(columns = ['Name','Ticket','Cabin','SibSp','PassengerId','Parch','Pclass','Embarked'])
encoded_test = pd.get_dummies(test, columns=['Sex'])
encoded_test = encoded_test.fillna(encoded_test.median())

predict = rfc.predict(encoded_test)
test_df['Survived'] = predict

output = test_df[['PassengerId','Survived']].to_csv('output.csv',index=False)
