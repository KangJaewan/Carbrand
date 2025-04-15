import cv2
import os
import supervision as sv
from ultralytics import YOLO

# YOLOv10 객체 탐지 모델을 yolov10x.pt 가중치 파일로부터 로드
model = YOLO("yolo11n.pt")

# 객체 라벨링을 위한 사전
category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def process_webcam():
    # 기본 웹캠에 접근하여 비디오 스트림을 읽습니다
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
    
    # 웹캠을 열지 못한 경우 오류 메시지를 출력하고 함수를 종료
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 객체 감지 결과를 시각화할 수 있도록 경계 상자와 라벨을 추가하는 도구인 
    # BoundingBoxAnnotator와 LabelAnnotator를 초기화합니다.
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        # 웹캠으로부터 비디오 프레임을 읽어옴 
        # ret은 프레임이 제대로 읽혔는지 여부를 나타내고, 
        # frame은 실제 이미지 데이터를 가집니다.
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)[0]
        # YOLO 모델의 감지 결과를 sv.Detections 객체로 변환하여 후처리하기 쉽게 만듭니다.
        detections = sv.Detections.from_ultralytics(results)
        # 감지된 객체마다 경계 상자(cv2.rectangle)와 객체 이름 및 신뢰도(cv2.putText)를 프레임에 그립니다. 
        # 클래스 ID는 category_dict를 사용하여 객체의 이름(예: person)으로 변환됩니다.
        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict[class_id]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()
