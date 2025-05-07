import cv2
from deepface import DeepFace # type: ignore

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotions = result[0]['emotion']
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    dominant_emotion, percentage = sorted_emotions[0]
    
    cv2.putText(frame, f"{dominant_emotion}: {percentage:.2f}%", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
