from ultralytics import YOLO
import cv2

#this is the path to the trained YOLO model weights (.pt file).
#these weights were produced after training and contain everything the model learned.
WEIGHTS = r"runs\detect\train\weights\best.pt"

#running on CPU mode (i do not have CUDA or GPU support).
DEVICE = "cpu"

#confidence threshold determines how sure a prediction must be to appear.
#lower = more detections but more false positives.
#higher = fewer detections but more accuracy.
CONF = 0.3
IMG_SIZE = 640
SKIP = 2

#load the trained model from the weights file.
model = YOLO(WEIGHTS)

#capturing my web camera (only 1)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#if no camera then exit
if not cap.isOpened():
  print("Could not open webcam...")
  exit()

#set webcam resolution for display clarity
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#improves efficiency and processing speed.
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

frameID = 0
lastAnnotated = None

#will read a new camera frame if fails then exits loops
while True:
  #frame will be the image of the detection and ret will determine if its being read (the camera)
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame...")
    break

  frameID += 1
  if frameID % SKIP == 0 or lastAnnotated is None:
    #testing the model in real time (camera)
    results = model.predict(source=frame, conf=CONF, device=DEVICE, imgsz=IMG_SIZE,verbose=False)
    #drawing the detections on the frame
    annotated = results[0].plot()

  #the output
  cv2.imshow("YOLOv8 - Mask Detector",annotated)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
#cleaning up so there are no leaks
cap.release()
cv2.destroyAllWindows()