import os
import shutil
from os import path
import random
import time

PATH = "/home/demir/Desktop/RTUK/Data/Images/ClassC/"
Out = "/home/demir/Desktop/RTUK/Data/Images/ClassCC/"
count = 0
ls = []

for i in os.listdir(PATH):
	ls.append(i)

random.shuffle(ls)

for i in ls:

	if i.split(".")[1]=="jpg":

		shutil.copyfile(PATH+i,Out+'{}.jpg'.format(count))
		shutil.copyfile(PATH+i.split(".")[0]+".xml",Out+'{}.xml'.format(count))
		count+=1
