from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading
import json
from gtts import gTTS
import os
import pygame
import time
from app import generate_fire_message  # GPT 기반 멘트 생성 함수 불러오기

app = Flask(__name__)

# 모델과 비디오 불러오기
model = YOLO("yolov11s_5.pt")
# model = YOLO("v3.pt")
video_path = "fire_video1.mp4"
cap = cv2.VideoCapture(video_path)

# 클래스 이름 불러오기
CLASS_NAMES = model.names.values()


# 상태 및 알람 재생 여부
last_detection = "정상"
alarm_playing = False

def play_alarm_and_tts(message):
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True

        # TTS 생성
        tts = gTTS(text=message, lang='ko')
        tts.save("static/tts.mp3")

        pygame.mixer.init()

        # 1차 알람
        pygame.mixer.music.load("static/alarm.mp3")
        pygame.mixer.music.play()
        time.sleep(5)
        pygame.mixer.music.stop()

        # 멘트 재생
        pygame.mixer.music.load("static/tts.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)

        # 2차 알람
        pygame.mixer.music.load("static/alarm.mp3")
        pygame.mixer.music.play()
        time.sleep(10)
        pygame.mixer.music.stop()

        # 정리
        if os.path.exists("static/tts.mp3"):
            os.remove("static/tts.mp3")

        alarm_playing = False


#  YOLO 감지 처리
def generate():
    global last_detection
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        detected_labels = []
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]

            if label == "Fire" and conf >= 0.55:
                detected_labels.append("Fire")
            elif label == "Smoke" and conf >= 0.6:
                detected_labels.append("Smoke")

        if "Fire" in detected_labels:
            last_detection = "🔥 화재 발생!"
            threading.Thread(target=handle_detection, args=("Fire",)).start()
            print(f"[ALERT] 🔥 화재 감지됨!  ")

        elif "Smoke" in detected_labels:
            last_detection = "🌫️ 연기 감지됨"
            threading.Thread(target=handle_detection, args=("Smoke",)).start()
            print("[NOTICE] 🌫️ 연기 감지됨. 주의하세요.")


        else:
            last_detection = "정상"

        # 시각화 및 스트리밍
        annotated = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def handle_detection(kind):
    print(f"[DEBUG] handle_detection 호출됨: {kind}")
    try:
        if kind == "Fire":
            message = generate_fire_message()
        else:
            message = generate_fire_message()
    except Exception as e:
        print(f"[ OPEN API 토큰부족]: {e}")
        #  GPT(토큰부족시) 기본 메시지로 대체
        if kind == "Fire":
            message = "화재가 감지되었습니다. 화재가 감지되었습니다. 신속히 대피해주십시오. 화재가 감지되었습니다. 화재가 감지되었습니다. 신속히 비상계단을 이용해주십시오."
        else:
            message = "연기가 감지되었습니다. 주의하십시오. 연기가 감지되었습니다. 주의하십시오."

    # 알람은 재생되도록 호출
    play_alarm_and_tts(message)



@app.route('/')
def index():
    return render_template('index.html', status=last_detection)

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return last_detection

if __name__ == "__main__":
    app.run(debug=True, port=8080)
