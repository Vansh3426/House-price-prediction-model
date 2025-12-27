import numpy as np 
import pandas as pd 
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler



def rmse(t,p):
  return np.sqrt(np.mean(np.square(t-p)))

df = pd.read_csv('kc_house_data.csv')

inputs =df.drop(columns=["id","price","date"])
target =df['price']


print(inputs.shape)
print(target.shape)

scaler = StandardScaler()
# scalerT = StandardScaler()

inputs_scaled = scaler.fit_transform(inputs)
# target_scaled = scalerT.fit_transform(target)

model = SGDRegressor(max_iter=2000, learning_rate="constant", eta0=0.0001)
train_model = model.fit(inputs_scaled, target)

example =np.array([[3,2,1680,8080,1,0,0,3,8,1680,0,1987,0,98074,47.6168,-122.045,1800,7503]])
# example1 =inputs
example_scaled = scaler.transform(example)
predictions = train_model.predict(example_scaled)

print(predictions)

# predf =pd.DataFrame(predictions) 
# print(predf.iloc[0:10])

# loss = rmse(target,predictions)

# print("your loss is ", loss)

