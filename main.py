import sys
import time
import threading
import cv2  # type: ignore
import dlib  # type: ignore
import numpy as np  # type: ignore
import pygame  # type: ignore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QMessageBox, QGraphicsDropShadowEffect
)

pygame.mixer.init()


class DetectionThread(threading.Thread):
    def __init__(self, update_frame, update_status, cooldown_update):
        super().__init__()
        self.update_frame = update_frame
        self.update_status = update_status
        self.cooldown_update = cooldown_update
        self.running = True
        self.cooldown = False
        self.cooldown_start = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.cap = cv2.VideoCapture(0)
        self.alarm_file = "alarm.wav"
        self.ear_threshold = 0.21
        self.frame_threshold = 15
        self.low_ear_counter = 0

    def run(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.resize(frame, (480, 360))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                reason = ""

                if faces:
                    for face in faces:
                        landmarks = self.predictor(gray, face)
                        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
                        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
                        left_ear = self.eye_aspect_ratio(left_eye)
                        right_ear = self.eye_aspect_ratio(right_eye)
                        ear = (left_ear + right_ear) / 2.0
                        angle = self.head_tilt_angle(landmarks)

                        if ear < self.ear_threshold:
                            self.low_ear_counter += 1
                        else:
                            self.low_ear_counter = 0

                        if self.low_ear_counter >= self.frame_threshold:
                            reason += "Eyes Closed "
                        if angle > 25:
                            reason += "Head Tilted"

                        if reason.strip():
                            if not self.cooldown:
                                self.update_status(f"Drowsy Detected! 💩 ({reason.strip()})", "red")
                                self.start_alarm()
                                self.cooldown = True
                                self.cooldown_start = time.time()
                        else:
                            if not self.cooldown:
                                self.update_status("Active 🙂", "green")

                if self.cooldown:
                    remaining = 30 - int(time.time() - self.cooldown_start)
                    if remaining > 0:
                        self.cooldown_update(remaining)
                    else:
                        self.cooldown = False
                        self.cooldown_update(0)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                if self.running:
                    self.update_frame(pixmap)

                time.sleep(0.05)
        except Exception as e:
            print(f"[Thread Error] {e}")

    def start_alarm(self):
        try:
            pygame.mixer.music.load(self.alarm_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Failed to play alarm: {e}")

    def stop(self):
        self.running = False
        time.sleep(0.2)  # slight delay for thread to exit safely
        if self.cap.isOpened():
            self.cap.release()
        pygame.mixer.music.stop()

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    def head_tilt_angle(self, landmarks):
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return abs(np.degrees(np.arctan2(dy, dx)))


class DrowsinessDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.detector_thread = None
        self.latest_frame = None
        self.image_update_timer = QTimer()
        self.image_update_timer.timeout.connect(self.update_gui_image)

    def init_ui(self):
        self.setWindowTitle("Drowsiness Detection System")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            background-color: #2b2b2b;
            color: white;
        """)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("""
            background-color: #1e1e1e;
            border-radius: 15px;
        """)
        self.add_shadow(self.video_label)

        self.status_label = QLabel("Status: Not Started")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white;")

        self.cooldown_label = QLabel("")
        self.cooldown_label.setFont(QFont("Arial", 14))
        self.cooldown_label.setAlignment(Qt.AlignCenter)
        self.cooldown_label.setStyleSheet("color: lightgray;")

        self.start_button = QPushButton("Start Detection")
        self.start_button.setFont(QFont("Arial", 14))
        self.start_button.setFixedHeight(50)
        self.start_button.clicked.connect(self.toggle_detection)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.add_shadow(self.start_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.cooldown_label)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 0)
        shadow.setColor(Qt.black)
        widget.setGraphicsEffect(shadow)

    def toggle_detection(self):
        if self.detector_thread is None:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        self.detector_thread = DetectionThread(
            self.update_image, self.update_status, self.update_cooldown
        )
        self.detector_thread.start()
        self.image_update_timer.start(33)
        self.start_button.setText("Stop Detection")
        self.status_label.setText("Status: Running")

    def stop_detection(self):
        if self.detector_thread:
            self.detector_thread.stop()
            self.detector_thread.join()
            self.detector_thread = None
        self.image_update_timer.stop()
        self.start_button.setText("Start Detection")
        self.status_label.setText("Status: Stopped")
        self.cooldown_label.setText("")
        self.video_label.clear()

    def update_image(self, pixmap):
        self.latest_frame = pixmap

    def update_gui_image(self):
        if self.latest_frame:
            self.video_label.setPixmap(self.latest_frame.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def update_status(self, text, color="white"):
        self.status_label.setText(f"Status: {text}")
        self.status_label.setStyleSheet(f"color: {color};")

    def update_cooldown(self, remaining):
        if remaining > 0:
            self.cooldown_label.setText(f"Cooldown: {remaining} sec")
        else:
            self.cooldown_label.setText("")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetectorApp()
    window.show()
    sys.exit(app.exec_())
