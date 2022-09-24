from xml.parsers.expat import model
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv('heart.csv')
print(df.head())

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=99)


algorithm= linear_model.LogisticRegression()

algorithm.fit(x_train,y_train)

#pred = algorithm.predict([[58,0,0,100,248,0,0,122,0,1,1,0,2]])
#print(pred)

if pred==1:
    print("person has heart des")
else:
    print("good")

newpred = algorithm.predict(x_test)
acc= accuracy_score(y_test, newpred)
print("the accuracy of model",acc*100,"%")

pickle.dump(algorithm,open("model.pkl","wb"))

mp=pickle.load(open('model.pkl','rb'))
pred=mp.predict([[52,1,0,125,212,0,1,168,0,1,2,2,3]])
print(pred)