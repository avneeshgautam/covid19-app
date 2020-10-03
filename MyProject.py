import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("data.csv")

## training and testing
X = df.drop('infectionProb',axis=1)
y= df['infectionProb']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=51)

 
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)

file = open("model.pkl","wb")
pickle.dump(classifier,file)



file.close()