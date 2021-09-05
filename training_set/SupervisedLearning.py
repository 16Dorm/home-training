import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


#데이터 전처리 및 가공
df1=pd.read_csv("./training_set/train.csv", names=["head","shoulder","elbow","hand","hip","foot","angle","path","label"])
df2=df1.drop(columns=['path'])
x=df2.drop(columns=['label'])
y=df2['label']

#데이터 전처리(분할)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=2020)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#학습 
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

#평가
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)

print(classification_report(y_train,y_train_pred))