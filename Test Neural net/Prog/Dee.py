from keras.models import Sequential
import keras
from keras import optimizers
import numbers
import decimal
from keras.layers import Dense
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import model_selection
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
print(xa[9])
x,y = xa.shape
xaxa=xa
ya = converta(data.as_matrix(columns=['INCOME']))
yaya=ya
acc= pd.DataFrame([[0,1]],columns=['NUM','ACC'])
#defining model

print xa.shape
print ya.shape
test_siz=0.01
ok=1
seed = 7
dec=0.0
lra=0.01
b1=0.0
b2=0.01
ep=1e-08

while test_siz <=.99:
	while lra <=.99:
		while b1 <=0.9:
			while b2 <=0.99:
				ADAM = keras.optimizers.Adam(lr=lra, beta_1=b1, beta_2=b2, epsilon=1e-08, decay=0.0)
				xa=xaxa
				ya=yaya
				xa, xaa, ya, yaa = model_selection.train_test_split(xa,ya,test_size=test_siz, random_state = seed)
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
				model.compile(loss='mean_squared_logarithmic_error',optimizer=ADAM,metrics=['accuracy'])
				model.fit(xa,ya,epochs=5, batch_size=3000, verbose=1)
				#model.save("model_model.h5")
				#model = keras.models.load_model("model_model.h5")
				predictions=model.predict(xaa)
				scores=model.evaluate(xa,ya)
				print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
				lol=pd.DataFrame([[ok,(scores[1]*100)]], columns=['NUM','ACC'])
				acc = acc.append(lol)
				ok+=1
				b2+=.01
			b1+=0.1
		lr+=.01
	test_siz+=.01
gn=acc.as_matrix()

plt.scatter(gn[:,0], gn[:,1], color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.axis([0.0,50,74,78])
ax=plt.gca()
ax.set_autoscale_on(False)
plt.savefig('Neural.png', bbox_inches='tight')
plt.show()
