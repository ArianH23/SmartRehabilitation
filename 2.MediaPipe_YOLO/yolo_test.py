from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3, imgsz=540)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
results = model.predict("frames/precision/precision6784.png", show=True) # # predict on an image
while(True):
    pass
# success = model.export(format="onnx")  # export the model to ONNX format