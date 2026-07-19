import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import pickle 
import os

def clean_text(text):
    text=text.lower()
    text= re.sub(r"http\S+","", text)
    text= re.sub(r"[^a-zA-Z\s]","", text)
    return text
    

fake = pd.read_csv("data/Fake.csv")
true= pd.read_csv("data/true.csv")

fake["label"]=0
true["label"]=1

data=pd.concat([fake,true], ignore_index=True)
data=data.sample(frac=1,random_state=42).reset_index(drop=True)
data["text"]=data["text"].apply(clean_text)
data=data[["text", "label"]]

X=data["text"]
y=data["label"]
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)
model= LogisticRegression()
model.fit(X_train,y_train)
prediction= model.predict(X_test)

print("Accuracy:" , accuracy_score(y_test,prediction))
os.makedirs("model", exist_ok=True)

pickle.dump(model,open("model/model.pkl","wb"))



print("Model saved successfully!")
print(X_train.shape)
print(X_test.shape)
