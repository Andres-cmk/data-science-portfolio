# 📊 Data Science & Machine Learning Portfolio
 
Hi, I'm **Andrés Ramírez**, a Systems and Computer Engineering student at **Universidad Nacional de Colombia**. This repository collects my data science, machine learning and deep learning projects. Each project has its own folder with a notebook, a README and its requirements.

---
 
## 🚀 Projects
 
| Project | Problem | Techniques | Key result |
|---|---|---|---|
| [💳 Fraud Detection](./fraudDetection) | Detect fraudulent mobile money transactions (0.13% fraud rate) | scikit-learn pipelines, Logistic Regression, Random Forest, CRISP-DM | **F1 85.9%** (precision 95.7%, recall 77.9%) |
| [🔊 Scream Detection with CNN](./cnnAudio) | Classify audio clips as scream or no scream | PyTorch, torchaudio, Mel spectrograms, CNN | **78% accuracy**, F1 0.73 on the scream class |
| [🚗 Vehicle Collision Detection](https://github.com/Andres-cmk/Fraud-Detection-System) | Detect traffic accidents in images and video | YOLOv11, Roboflow, Python | Real-time inference on images and video |
| 🚧 Customer Churn Prediction | Predict which customers are likely to leave | In progress | Coming soon |
 
> Notebook explanations are written in Spanish. Every project README is in English.
 
---
 
## 🧰 What you'll find here
 
- **Classification on imbalanced data**, with metrics that go beyond accuracy (precision, recall, F1).
- **Deep learning** for audio and computer vision with PyTorch and YOLO.
- **Clean, reproducible pipelines** with scikit-learn, including preprocessing, modeling and model export.
- **Business-oriented analysis**: each project starts from a real problem and explains what the metrics mean for it.
---
 
## 🛠️ Tech stack
 
**Python** · **pandas** · **NumPy** · **scikit-learn** · **PyTorch** · **torchaudio** · **YOLOv11** · **Matplotlib** · **seaborn** · **plotly** · **Google Colab** · **Git**
 
---
 
## 📁 Repository structure
 
```
data-science-portfolio/
├── README.md
├── fraudDection/
│   ├── README.md
│   ├── Fraud_Detection_System.ipynb
│   └── requirements.txt
├── CnnAudio/
│   ├── README.md
│   ├── CNN_AUDIO.ipynb
│   └── requirements.txt
└── vehicle-collision-detection/
    ├── README.md
    └── ...
```
 
---
 
## ▶️ How to run a project
 
Every notebook opens in Google Colab with one click (the badge is at the top of each project README). To run one locally:
 
```bash
git clone https://github.com/Andres-cmk/data-science-portfolio.git
cd data-science-portfolio/fraud-detection
pip install -r requirements.txt
jupyter notebook
```
 
Datasets are not stored in this repository. Each project README links to its source (Kaggle) and explains how to download it.
 
---
 
## 🧱 Other projects
 
Backend and AI systems work lives in separate repositories:
 
- [UNAL Academic Chatbot](#): a chatbot that routes requests across several AI providers and answers with RAG over an institutional knowledge base (FastAPI, Next.js, Docker).
---
 
## 🌱 Currently learning
 
Deeper deep learning (recurrent networks, multi-input and multi-output architectures), MLOps and technical English.
 
---
 
## 📫 Contact
 
- LinkedIn: [Andrés Ramírez](https://www.linkedin.com/in/andres-felipe-ramirez-m/)
- GitHub: [Andres-cmk](https://github.com/Andres-cmk)
- Email: ramirezmontanaandresfelipe@gmail.com
 