import numpy as np
import argparse
import imutils
import time
import cv2
import os
from imutils.video import FPS
from imutils.video import VideoStream
from scipy.spatial import distance as dist


RTSP_URL = "people.mp4"
YOLO_PATH="yolo-coco"
OUTPUT_FILE="output/outfile.avi"

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
CONFIDENCE=0.5
THRESHOLD=0.3
MIN_DISTANCE = 100 # threshold social distance in pixels

# USE_GPU = True

# colors for each label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

# loading YOLO object detector
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream
vs = cv2.VideoCapture(RTSP_URL)
time.sleep(2.0)
fps = FPS().start()
writer = None
(W, H) = (None, None)

cnt=0



# loop over frames
while True:
	cnt+=1
	# read the next frame
	(grabbed, frame) = vs.read()

	results = []

	
	if not grabbed:
		break
	
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, centroids, confidences, and class IDs, respectively
	boxes = []
	centroids = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			
			if confidence > CONFIDENCE and classID==0:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
		THRESHOLD)

	
	if len(idxs) > 0:
		
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			r = (confidences[i], (x, y, x+w, y+h), centroids[i])
			results.append(r)

	violate = set()

        
	if len(results) >= 2:
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two centroid pairs is less than the configured number of pixels
				if D[i, j] < MIN_DISTANCE:
					# update our violation set 
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)
		
        
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		

	
	writer.write(frame)
	cv2.imshow("Frame", cv2.resize(frame, (800, 600)))
	key = cv2.waitKey(1) & 0xFF
	#print ("key", key)
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()


fps.stop()


print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
print("[INFO] cleaning up...")
writer.release()
vs.release()
