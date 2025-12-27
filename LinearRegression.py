import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression

def rmse(t,p):
  return np.sqrt(np.mean(np.square(t-p)))

df = pd.read_csv('kc_house_data.csv')

inputs =df.drop(columns=["id","price","date"])
target =df['price']


print(inputs.shape)
print(inputs.dtypes)
print(target.shape)

model = LinearRegression()

train_model  = model.fit(inputs,target)
 

example =np.array([[3,2,1680,8080,1,0,0,3,8,1680,0,1987,0,98074,47.6168,-122.045,1800,7503]])

predictions = train_model.predict(example)
print(predictions)



# loss = rmse(target,predictions

# print("your loss is ", loss)

