from ultralytics import YOLO
import cv2

model = YOLO("last.pt")

demo = "video/Knot_Tying_B002_capture1.mp4"

results = model.predict(source=demo, show=True)



print(results)