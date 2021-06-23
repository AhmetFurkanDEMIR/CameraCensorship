import cv2
import numpy as np
import skvideo.io

video_path = "/home/demir/Desktop/a/Data/Video/ClassC/WIN_20210621_13_38_34_Pro.mp4"
out_path = "outpy.avi"

video_capture = cv2.VideoCapture(video_path)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

fourcc = cv2.VideoWriter_fourcc(*'H264')
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (frame_width, frame_height))
video_capture = cv2.VideoCapture(video_path)

while True:

	flag, frame = video_capture.read()

	if flag==False:

		break

	else:

		w = frame.shape[0]
		h = frame.shape[1]

		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (w, h)), 1.0,
		(w, h), (104.0, 177.0, 123.0))


		net.setInput(blob)
		detections = net.forward()

		for i in range(0, detections.shape[2]):


			confidence = detections[0, 0, i, 2]


			if confidence < 0.5:
				continue


			box = detections[0, 0, i, 3:7] * np.array([h, w, h, w])
			(startX, startY, endX, endY) = box.astype("int")


			frame[startY:endY, startX:endX] = (0,0,0)

		frame=frame.reshape(w,h,3)
	out.write(frame)

out.release()
video_capture.release()