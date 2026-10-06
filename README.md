# VisionQC

철강 표면 이미지를 기반으로 결함 종류와 위치를 자동으로 판별하고,
학습되지 않은 이상 패턴까지 감지할 수 있는 AI 품질검사 시스템을 구축한다.

## Project Goal

- 이미지 분류: 철강 표면 이미지의 결함 종류 판별
- 객체 탐지: 결함의 종류와 위치 판별
- 이상 탐지: 학습되지 않은 이상 패턴 감지
- 추론 파이프라인과 웹 애플리케이션으로 검사 결과 제공

현재는 Day 1 데이터셋 구조 탐색 단계이며, 모델 구현과 학습은 시작하지 않았다.

## Dataset

Day 1 탐색 대상은 **NEU-DET**이며, 탐색 코드는 프로젝트 루트의
`data/raw/NEU-DET/` 경로를 사용한다. `train`과 `validation` 각각에
`images/<클래스명>/`과 `annotations/*.xml` 폴더가 필요하다.
클래스는 `train/images`의 하위 폴더에서 자동으로 찾는다.

현재 로컬 데이터인 `data/raw/NEU-DET/`를 그대로 탐색한다.
전처리 데이터용 `data/processed/`와 외부 보조 데이터용 `data/external/`은 유지한다.

데이터셋과 학습 모델 파일은 Git에 커밋하지 않는다.
`.gitkeep`은 빈 폴더 구조를 유지하기 위한 파일이다.

## Project Structure

```text
visionqc/
├── data/
│   ├── raw/                 # 원본 데이터
│   │   └── NEU-DET/         # Day 1 탐색 대상
│   │       ├── train/       # images/<클래스명>/, annotations/*.xml
│   │       └── validation/  # images/<클래스명>/, annotations/*.xml
│   ├── processed/           # 전처리 데이터
│   └── external/            # 외부 보조 데이터
├── src/
│   ├── data/
│   │   └── explore_dataset.py
│   ├── models/
│   │   └── __init__.py
│   ├── training/
│   │   └── __init__.py
│   └── inference/
│       └── __init__.py
├── app/
│   └── __init__.py
├── models/                  # 학습 모델 저장
├── results/                 # 결과 저장
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

빈 폴더인 `data/raw`, `data/processed`, `data/external`, `models`, `results`에는
`.gitkeep` 파일을 추가했다.

## Development Environment

- IDE: VS Code
- Python: 3.11 이상, 로컬 실행
- Python `.py` 파일 중심 개발 (Notebook/Colab 사용하지 않음)
- 가상환경: `.venv`
- 버전관리: Git/GitHub 예정
- Day 1 라이브러리: numpy, pandas, matplotlib, Pillow
- TensorFlow, PyTorch, Ultralytics는 필요한 단계에서 추가 예정

### Windows PowerShell 시작 명령어

압축을 해제한 상위 폴더에서 실행한다. Python 3.11과 VS Code가 설치되어 있어야 한다.

```powershell
cd .\visionqc
code .
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
python src/data/explore_dataset.py
```

VS Code에서 `Python: Select Interpreter`를 실행하고
`.venv\Scripts\python.exe`를 선택한다.

PowerShell 실행 정책으로 활성화가 차단되면, 활성화 없이 다음 명령어를 사용할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe src/data/explore_dataset.py
```

`main.py`는 프로젝트 소개 메시지를 출력한다.
`explore_dataset.py`는 파일 위치를 기준으로 프로젝트 루트를 찾으므로
실행 위치에 관계없이 `data/raw/NEU-DET` 경로를 확인한다. 필수 경로가 없으면
없는 경로를 모두 출력한다. 정상 구조에서는 클래스별/전체 이미지 수,
실제 확장자, 첫 샘플의 크기와 모드, XML 개수와 split별 개수 일치 여부를 출력한다.
이미지는 `.jpg`, `.jpeg`, `.png`, `.bmp`를 대소문자 구분 없이 지원한다.
각 클래스 폴더와 annotation 폴더 바로 아래의 파일만 센다.
샘플 확인에는 Pillow를 사용한다. split별로 확장자를 제외한 파일명(stem)을
대소문자 구분하여 비교하고, 상대 파일이 없는 이미지/XML의 전체 개수와
각 목록의 처음 10개 파일명을 출력한다. XML 내용 파싱이나 YOLO 변환은 수행하지 않는다.
stem 일치는 XML 내용의 정확성이나 일대일 대응을 보장하지 않는다.

## Roadmap

1. Dataset Exploration
2. CNN Baseline
3. Transfer Learning
4. Fine-tuning
5. YOLO Object Detection
6. Autoencoder Anomaly Detection
7. Inference Pipeline
8. Web Application
