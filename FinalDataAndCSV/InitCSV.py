import pandas as pd
import os
import shutil
from os import path
import random
import time
from sklearn.utils import shuffle

PATH = "/home/demir/Downloads/data/training_images/C/train/augmented_imgs/"
Out = "/home/demir/Desktop/RTUK/Final/TrainFinal/"
count = 11401
main_csv = pd.read_csv("/home/demir/Desktop/RTUK/Final/FinalCSV/mainTrain.csv")
csv_temp = pd.read_csv("train_labels.csv")


for i in os.listdir(PATH):

	if i.split(".")[1]=="jpg":

		try:


			temp_data = list(csv_temp["filename"]).index(i)

		except:

			continue

		shutil.copyfile(PATH+i,Out+'{}.jpg'.format(count))
		data = pd.DataFrame({"filename":str(count)+".jpg", 
					"width":[csv_temp["width"][temp_data]], 
					"height":[csv_temp["height"][temp_data]], 
					"class":[csv_temp["class"][temp_data]], 
					"xmin":[csv_temp["xmin"][temp_data]], 
					"ymin":[csv_temp["ymin"][temp_data]], 
					"xmax":[csv_temp["xmax"][temp_data]], 
					"ymax":[csv_temp["ymax"][temp_data]]})

		count+=1
		main_csv = main_csv.append(data)

main_csv = shuffle(main_csv)
main_csv.to_csv("/home/demir/Desktop/RTUK/Final/FinalCSV/mainTrain.csv", index=False)