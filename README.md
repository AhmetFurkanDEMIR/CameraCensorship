 ![](https://img.shields.io/badge/microsoft%20azure-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white) ![](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) ![](https://img.shields.io/badge/NVIDIA-Tesla%20K80-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white) ![](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white)
 
# **CameraCensorship**

* With the TensorFlow 2 Object Detection API, unwanted, unethical images are detected and censored in the camera.

* There are also 3 classes in total, the first and second classes are the undesirable immoral classes, and the third class is the positive class. Data reserved for training purposes 1.5 GB (train.record), data reserved for testing purposes 516 MB (test.record).

* The training is based on the faster_rcnn_resnet101_v1_640x640_coco17 model.

* The training was trained with Nvidia Tesla K80 GPU via Microsoft Azure.


## Data, Training and Test steps

**Data :** First, we collected data for training, we shot videos in different environments with different t-shirts, then we covered the faces in this video with a black mask, then we divided these videos into .jpg files step by step and collected the data. We specified the classes in the image with the labeling operation on this data, then we made duplication on these tagged data, and finally we converted the tag information of this data into .csv file and combined the tags and images to create .record files.

**Data Helper files**

* [Face masking](/FaceCensorshipInData/)

* [Splitting videos into image files step by step](/FinalDataAndCSV/SplitVideoToPictures.py)

* [Data Augmentation](https://github.com/omerfarukkkoc/data_augmentation_for_labeled_images)

* [Mixing and renaming data](/FinalDataAndCSV/DataShuffle.py)

* [Data labeling](https://github.com/tzutalin/labelImg)

* [Creating a .record file by combining data and .csv files](/FinalDataAndCSV/generate_tfrecord.py)

**Train and Test :** After making the necessary configurations in TensorFlow 2 Object Detection API, training is started with a powerful gpu, you can monitor all the results of your training with tensorboard. At any time, you can end the training and get the trained weights, then you can test with these weights.

**Train and Test Helper files**

* TensorFlow 2 Object Detection API [Link1](https://github.com/tensorflow/models), [Link2](https://github.com/AhmetFurkanDEMIR/AI-Talent-Programme-2)

* [File to view test results](/RunModel/TestModel.py)


## Test results

![A](https://user-images.githubusercontent.com/54184905/123063638-46600900-d416-11eb-8426-f0b996e55064.png)

![B](https://user-images.githubusercontent.com/54184905/123063650-47913600-d416-11eb-9b4f-7835bbee0bb8.png)

