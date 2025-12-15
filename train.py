from ultralytics import YOLO

#train a model from scratch
model = YOLO("yolov8n.yaml")

#going to train from the data.yaml where i call the information of the classes
results = model.train(data="data.yaml",epochs=200,imgsz=640,batch=16)