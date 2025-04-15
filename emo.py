from ultralytics import YOLO
import cv2

# YOLO 모델 불러오기
model = YOLO("/Users/kangjaewan/code/gitproject/YOLOTEST/Carbrand/best.pt")

# 웹캠 열기 (0번은 기본 카메라)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델에 프레임 입력 후 결과 추론
    results = model(frame)

    # 결과 프레임 시각화 (결과[0].plot() 사용)
    result_frame = results[0].plot()

    # 출력
    cv2.imshow("/Users/kangjaewan/code/gitproject/YOLOTEST/Carbrand/best.pt Detection", result_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
