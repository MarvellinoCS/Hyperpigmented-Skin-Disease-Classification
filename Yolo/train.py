from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='./4disease_new', epochs=50, imgsz=224)
