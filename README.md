# Guess-Fist-Draw
This is a personal Computer Vision project about 2 tasks: Object detection and Classification. It allows users to draw by their fist when using webcam, and the program will guess which the user has just drawn. 

## Project structure
Guess-Fist-Draw/
│
├── data/
│   ├── fist/
│   │   ├── train/
│   │   └── valid/
│   └── draw/
│       ├── train/
│       └── valid/
│
├── models/
│   ├── fist/
│   │   ├── model.py        # fist detection model
│   │   ├── dataset.py
│   │   └── train.py
│   │
│   └── draw/
│       ├── model.py        # draw classification model
│       ├── dataset.py
│       └── train.py
│
├── inference/
│   ├── fist_detector.py
│   └── draw_classifier.py
│
├── utils/
│   ├── webcam.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── notebooks/
│   ├── train_fist.ipynb
│   ├── train_draw.ipynb
│   └── demo.ipynb
│
├── main.py                 # chạy webcam realtime
├── requirements.txt
└── README.md

## Set up
conda create -n torchcuda python=3.10 -y
conda activate torchcuda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -r requirements.txt