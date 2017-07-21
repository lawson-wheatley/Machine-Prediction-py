#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  
#  Copyright 2017 LWHEATLEY <lwheatley@lin24.ad.csbsju.edu>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
from sklearn import model_selection
from sklearn import linear_model
from sklearn.externals import joblib
import pickle
import matplotlib.pyplot as plt
import numbers
import decimal
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
np.random.seed(8)
'''
nmapa={}

def converta(mtrx):
	a=0
	c=1

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
	'''
data = pd.read_csv("PimaB.csv",index_col=0,delimiter=',')
data2 = pd.read_csv("PimaA.csv",index_col=0)
#SPLIT

#xa = data[['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY']]
xa=data.as_matrix(columns=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY'])
xaa = data2.as_matrix(columns=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY'])
xaaa = xaa
xaaaa= xaa
xaaaaa=xaa
loc=0
ya = data.as_matrix(columns=['INCOME'])
yaa = data2.as_matrix(columns=['INCOME'])
xa=xa[:,:]
ya=ya[:,0]
test_siz = 0.4
seed = 10

#xa, xaa, ya, yaa = model_selection.train_test_split(xa,ya,test_size=test_siz, random_state = seed)

model=linear_model.LinearRegression()
model.fit(xa, ya)
filename='Model_model.save'
joblib.dump(model, filename)

loaded_model=joblib.load(filename)
#result=loaded_model.score(xaa,yaa)
#print(result)
mm=0
aom=model.predict(xaa)
print("Mean error NR: %.2f" % np.mean((aom - yaa)))
aok=np.round(model.predict(xaa))
print("Mean error R: %.2f" % np.mean((aok - yaa)))
x,y = xaa.shape
g=0.00
for c in range(0,x):
	if aok[c] ==2:
		aok[c] =1
	if (aok[c] != yaa[c]):
		g+=1.00
	if (c%3000 ==0):
		print("Accuracy:"+str((float((x-g))/x)*100)+"%")
print("Final Accuracy:"+str((float((x-g))/x)*100)+"%")
plt.scatter(xaa[:,mm], aok, color='blue', linewidth=5)
plt.scatter(xaa[:,mm], yaa,  color='black')

plt.xticks(())
plt.yticks(())

plt.show()
