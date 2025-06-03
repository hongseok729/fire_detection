#  Fire Detection Model & Realtime Alert System

YOLOv11 기반의 화재·연기 감지 AI 모델을 학습하고, 실시간 감지 결과에 따라 긴급 알림 방송(TTS)을 생성하는 자동화 시스템입니다.  
GPT-4를 이용해 감지 상황에 맞는 안전 안내 멘트를 자동으로 생성하고, Flask 웹 서버를 통해 실시간 시각화 및 경고음 알림을 제공합니다.


fire_project/
├── app.py # OpenAI 기반 GPT 멘트 생성
├── main.py # 실시간 감지 Flask 서버
├── yolov11s.pt # YOLOv11 사전학습 모델
├── yolov11s_5.pt # Fire/Smoke 재학습 모델
├── static/
│ ├── style.css
│ ├── tts.mp3 # 동적 생성 TTS 파일
│ └── alarm.mp3 # 알람 사운드
├── templates/
│ └── index.html # 웹 인터페이스
└── .gitignore # 민감 파일 무시 설정



---

##  1. 학습 모델 구성 (YOLOv11 + Albumentations)

###  설치 및 데이터 전처리

```python
!git clone https://github.com/ultralytics/ultralytics.git
%cd ultralytics
!pip install -e .
!pip install albumentations roboflow opencv-python

Roboflow 데이터 다운로드 & 증강

from roboflow import Roboflow
rf = Roboflow(api_key="")  # 환경변수 방식 권장
project = rf.workspace("").project("fire-smoke-detection")
dataset = project.version(1).download("yolov11")

Albumentations 기반 이미지 증강 + 저장

def preprocess_for_yolov11(src_dir, dst_dir):
    # HorizontalFlip, MotionBlur, Resize 등 적용
    # bbox 좌표까지 함께 변경

##  2. data.yaml 설정

train: /content/preprocessed_yolov11/train/images
val: /content/preprocessed_yolov11/valid/images
nc: 2
names: ["Fire", "Smoke"]


##  3. YOLOv11 모델 학습

from ultralytics import YOLO

model = YOLO("yolov11s.pt")  # 기존 모델 불러오기

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="fire_smoke_yolov11s"
)

##  4. 실시간 감지 웹앱 실행

python main.py

## GPT 기반 알림 멘트 생성 (app.py)

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_fire_message():
    prompt = """화재 감지 시스템이 다음 위치에서 이상 상황을 포착했습니다:
    안전 안내방송용 문장을 정중하고 긴박하게 1문장으로 만들어줘."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


<알람 재생 흐름>

감지 → 알림 종류 파악 (Fire / Smoke)

GPT-4로 문장 생성 → TTS 생성

알람.mp3 + 생성된 tts.mp3 순차 재생

재생 완료 후 TTS 파일 삭제

<향후 개선사항?

CCTV 실시간 연동 (USB / IP Cam)

공공 화재 경보 시스템 연계

멀티채널 다중 객체 감지 및 대응

