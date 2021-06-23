import cv2

video_path = "/home/demir/Desktop/a/Data/Video/ClassC/outpy2.avi"
out_path = "/home/demir/Desktop/a/SplitVideoToPictures/Images/ClassC/"

step = 5
count = 0
count_images = 155

video_capture = cv2.VideoCapture(video_path)

while True:

	flag, frame = video_capture.read()

	if flag==False:

		break

	else:

		if count==step:

			count=0

			cv2.imwrite('{}/{}.jpg'.format(out_path,count_images), frame)
			count_images+=1


	count+=1


