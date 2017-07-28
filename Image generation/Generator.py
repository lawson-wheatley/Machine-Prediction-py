#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Generator.py
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

from scipy import ndimage
from scipy import misc
from random import randint
import time
import matplotlib.pyplot as plt
import numpy as np

#img = np.zeros((1024,1024,3),dtype=np.uint8)
img = misc.face()
#np.resize(img,(480,640,3))
#a=1023#b=1023
#l=3
b,a,l=img.shape
b-=1
a-=1
print(img.shape)
#time.sleep(10)
g=0
while g in range(0,a):
	d=0
	g+=1
	print("Row:"+str(g))
	while d in range(0,b):
		c=0
		#print("R"+str(g)+"C"+str(d))
		d+=1
		while c in range(0,l):
			#print(img[g][d][c])
			mm=256-img[d][g][c]
			#randint(0,256)
			#print("R"+str(g)+"C"+str(d)+"COLOR"+str(mm))
			img[d][g][c]=mm
			c+=1
#img = ndimage.gaussian_filter(img, sigma=1)
print(type(img))
print(img.shape)
#np.resize(img,(x,y,z))
print(img.shape)
plt.imshow(img)
plt.show()
