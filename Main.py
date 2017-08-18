import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import optimizers
#All keras imports, including the layers, the model type, along with compatibility with other libraries and optimizers.
from matplotlib import pyplot as plt
#Matlab plot library for plotting items.
import pandas as pd
import numpy as np
#Dataframing, and matrix working
#from sklearn import model_selection
#Allows for selection of specific parts of the data for training, and for testing.
import numbers
import decimal
#Numbers in case of compatibility
#import time
import sys

#Sys for arguments (for files), time for making the program sleeping in case of errors with time.sleep(Time in seconds) being the use case
print ("Loading data...")
data = open(str(sys.argv[1])).read()
#As it says, this part is loading data, with sys.argv[1] saying that the first argument will be used for the file object location.
data = data.lower()
#Converting all the data to lowercase (makes it so that no upper case will be in the midst of other characters!)
datac = list(data)#sorted()
ndata=len(data)
nchar=len(datac)
print(np.unique(data).shape)
print("Num chars=" +str(ndata))
print("Vocab="+str(nchar))

seqlen=100
dataX=[]
dataY=[]
for i in range(0, nchar - seqlen,1):
	seqi=data[i:i+seqlen]
	seqo=data[i+seqlen]
	dataX.append([ord(char) for char in seqi])
	dataY.append(ord(seqo))
numpat=len(dataX)
X = np.reshape(dataX,(numpat,seqlen,1))
print(X.shape)
#X=X/float(nchar)
Y=np_utils.to_categorical(dataY)
print(Y.shape)
#Y = np.reshape(dataY,(numpat,1))

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='ADAM',metrics=['accuracy'])


filepath="Model.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list=[checkpoint]
model.fit(X,Y,epochs=4,batch_size=64,callbacks=callbacks_list)
model.save("Mdel.hdf5")

model = keras.models.load_model("Mdel.hdf5")
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
# generate characters
odata=[]
for i in range(0,1000):
	print "#",i," START"
	x = np.reshape(pattern, (1, len(pattern), 1))
	#x = x / float(nchar)
	prediction = model.predict(x, verbose=0)
	print(prediction.shape)
	index = np.argmax(prediction)
	print(index,":",prediction[0][index])
	result = chr(index)
	odata.append(result)
	#seq_in = [chr(value) for value in pattern]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
	print "#",i," DONE CHAR:'",result,"'"
print "Output:",''.join(str(odata[a]) for a in range(0,1000))
print "\nDone."

