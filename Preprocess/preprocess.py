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


import matplotlib.pyplot as plt
import numbers
import decimal
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
np.random.seed(8)
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
data = pd.read_csv("pima.csv",index_col=0,delimiter=',',)
data2 = pd.read_csv("pima2.csv",index_col=0)
#SPLIT

#xa = data[['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY']]
xa=converta(data.as_matrix(columns=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY', 'INCOME']))
xaa = converta(data2.as_matrix(columns=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION_NUM','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL_GAIN','CAPITAL_LOSS','HOURS_PER_WEEK','NATIVE_COUNTRY', 'INCOME']))
x,y=xa.shape
with open('Processed_DATA.csv','w') as f:
	f.write("INDEX,AGE,WORKCLASS,FNLWGT,EDUCATION,EDUCATION_NUM,MARITAL_STATUS,OCCUPATION,RELATIONSHIP,RACE,SEX,CAPITAL_GAIN,CAPITAL_LOSS,HOURS_PER_WEEK,NATIVE_COUNTRY,INCOME\n")
	for col in range(x):
		f.write(str(col+1)+","+str(xa[col][0])+","+str(xa[col][1])+","+str(xa[col][2])+","+str(xa[col][3])+","+str(xa[col][4])+","+str(xa[col][5])+","+str(xa[col][6])+","+str(xa[col][7])+","+str(xa[col][8])+","+str(xa[col][9])+","+str(xa[col][10])+","+str(xa[col][11])+","+str(xa[col][12])+","+str(xa[col][13])+","+str(xa[col][14])+"\n")
		print(str(col+1)+","+str(xa[col][0])+","+str(xa[col][1])+","+str(xa[col][2])+","+str(xa[col][3])+","+str(xa[col][4])+","+str(xa[col][5])+","+str(xa[col][6])+","+str(xa[col][7])+","+str(xa[col][8])+","+str(xa[col][9])+","+str(xa[col][10])+","+str(xa[col][11])+","+str(xa[col][12])+","+str(xa[col][13])+","+str(xa[col][14]))
	time.sleep(1)
	f.close()
