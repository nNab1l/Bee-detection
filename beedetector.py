import torch
import cv2
import numpy as np
import screeninfo 
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

class ObjectDetection:

    def __init__(self, image_path):
        self.image_path = image_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(self):
        model = YOLO("weights/newdata.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, confidence, class_id, tracker_id in detections]

        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)

        return frame

    def process_image(self):
        frame = cv2.imread(self.image_path)
        results = self.predict(frame)
        frame = self.plot_bboxes(results, frame)

        screen_width, screen_height = self.get_screen_resolution()

        frame = self.resize_image_to_fit_screen(frame, screen_width, screen_height)

        cv2.imshow('YOLOv8 Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_screen_resolution(self):
        screen_info = screeninfo.get_monitors()[0]  
        screen_width = screen_info.width
        screen_height = screen_info.height
        return screen_width, screen_height

    def resize_image_to_fit_screen(self, image, screen_width, screen_height):
        image_height, image_width, _ = image.shape

        aspect_ratio = min(screen_width / image_width, screen_height / image_height)

        new_width = int(image_width * aspect_ratio)
        new_height = int(image_height * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height))

        return resized_image

image_path = "img/bumbles.jpg"  
detector = ObjectDetection(image_path=image_path)
detector.process_image()
