from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/ultralytics-main/ultralytics-main/ultralytics/yolov8n.pt")

model.train(model="yolov8.yaml",data="/root/autodl-tmp/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/coco.yaml",epochs=100,imgsz=640)


