from keras.models import Sequential
import keras
from keras import optimizers
import numbers
import decimal
from keras.layers import Dense
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
np.random.seed(8)
nmapa={}
def converta(mtrx):
	a=0
	c=0

	nr,nc=mtrx.shape
	print nr
	for col in range(0,nc):
		c=0

		try:
			val= float(mtrx[0,col])
			#a+=1
		except ValueError:
			for row in range(0,nr):
				if mtrx[row,col] in nmapa:
					mtrx[row,col] = nmapa[mtrx[row,col]]
					#b+=1
				else:
					nmapa[mtrx[row,col]]=c
					mtrx[row,col]=c
					c+=1
					#b+=1
	return(mtrx)
data = pd.read_csv("pima.csv",index_col=0,delimiter=',',)
data2 = pd.read_csv("pima2.csv",index_col=0)
#SPLIT

#xa = data[['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY']]
xa=converta(data.as_matrix(columns=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY']))
xaa = converta(data2.as_matrix(columns=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY']))
xaaa = xaa
xaaaa= xaa
xaaaaa=xaa
print(xa[9])
x,y = xa.shape
print(x)
print(y)
ya = converta(data.as_matrix(columns=['INCOME']))
#defining model
model=Sequential()
#model.add(Dense(512, input_shape=(14,), activation='tanh'))
#model.add(Dense(256, activation = 'tanh'))
model.add(Dense(1792, input_shape=(14,),activation = 'softmax'))
model.add(Dense(448,activation = 'relu'))
model.add(Dense(224, activation = 'sigmoid'))
model.add(Dense(112, activation = 'relu'))
model.add(Dense(56, activation = 'relu'))
model.add(Dense(28, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='mean_squared_logarithmic_error',optimizer='SGD',metrics=['accuracy'])
print xa.shape
print ya.shape
model.fit(xa,ya,validation_split=0.1, verbose=1)
model.save("model_model.h5")
model = keras.models.load_model("model_model.h5")
print("Prediction 1")
predictions=model.predict(xaa)
scores=model.evaluate(xa,ya)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
rounded = [round(xaa[0]) for xaa in predictions]
print(rounded)
print("Prediction 2")
predictions=model.predict(xaaa)
scores=model.evaluate(xa,ya)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
rounded = [round(xaaa[0]) for xaaa in predictions]
print(rounded)
print("Prediction 3")
predictions=model.predict(xaaaa)
scores=model.evaluate(xa,ya)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
rounded = [round(xaaaa[0]) for xaaaa in predictions]
print(rounded)
print("Prediction 4")
predictions=model.predict(xaaaaa)
scores=model.evaluate(xa,ya)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
rounded = [round(xaaaaa[0]) for xaaaaa in predictions]
print(rounded)
#fig, ax = plt.subplots(1,1)
#ax[0].hist(xa, 10, facecolor='red', alpha=0.5, label= "Simulated")
#ax[0].hist(data[:,0:8], facecolor='black',alpha=0.5,label="Actual")
