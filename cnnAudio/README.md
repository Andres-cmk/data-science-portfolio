# 🔊 Scream Detection with Deep Learning and CNNs
 
Binary audio classifier that detects **human screams** by converting sound into **Mel spectrograms** and feeding them to a **convolutional neural network (CNN)** built in PyTorch.
 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andres-cmk/data-science-portfolio/blob/main/audio-scream-cnn/CNN_AUDIO.ipynb)
 
> The notebook's explanations (Fourier transform, Nyquist theorem, Mel scale) are written in Spanish. This README summarizes the project in English.
 
---
 
## 🎯 Goal
 
Given an audio clip, decide whether it contains a scream (`Screaming`) or not (`NotScreaming`). This kind of model can be a building block for safety and monitoring applications.
 
## 🧠 Approach
 
```
Audio clip → mono + resample to 16 kHz → fixed 10 s length → Mel spectrogram (log scale) → CNN → scream / no scream
```
 
1. **Preprocessing (torchaudio):** convert to mono, resample to 16 kHz and pad or trim every clip to 10 seconds.
2. **Features:** log-scaled Mel spectrograms (`n_fft=1024`, `hop_length=512`, `n_mels=64`), which produce about 313 time frames per clip. The spectrogram is treated as a one-channel image.
3. **Model:** a CNN with 4 convolutional blocks and a small classifier head.
4. **Training:** the model is trained with `BCEWithLogitsLoss` and the checkpoint with the lowest validation loss is saved.
### Architecture
 
| Stage | Layers |
|---|---|
| Block 1 | Conv2d (1 → 16) · BatchNorm · ReLU · MaxPool |
| Block 2 | Conv2d (16 → 32) · BatchNorm · ReLU · MaxPool |
| Block 3 | Conv2d (32 → 64) · BatchNorm · ReLU · MaxPool |
| Block 4 | Conv2d (64 → 128) · BatchNorm · ReLU · AdaptiveAvgPool |
| Classifier | Flatten · Dropout (0.3) · Linear (128 → 64) · ReLU · Linear (64 → 1) |
 
The adaptive pooling in the last block lets the network handle any spectrogram width.
 
## 📦 Dataset
 
[Human Screaming Detection Dataset](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset) (Kaggle). I used a **balanced subset of 1,000 clips** (500 `Screaming` and 500 `NotScreaming`), split **80% / 20%** into 800 training and 200 validation samples.
 
## ⚙️ Training setup
 
| Parameter | Value |
|---|---|
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-2`) |
| Loss | BCEWithLogitsLoss |
| Batch size | 32 |
| Epochs | 20 |
| Checkpoint | Lowest validation loss (`best_model.pth`) |
 
## 📊 Results
 
Evaluated on the 200 validation clips:
 
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NotScreaming | 0.78 | 0.86 | 0.82 | 113 |
| Screaming | 0.79 | 0.68 | 0.73 | 87 |
| **Accuracy** | | | **0.78** | 200 |
 
The model is better at recognizing non-screams than screams: it misses about one in three screams (recall 0.68).
 
<!-- Add your training curves and confusion matrix here, for example:
![Training curves](images/training_curves.png)
![Confusion matrix](images/confusion_matrix.png)
-->
 
## ⚠️ Limitations
 
- **Small dataset:** only 1,000 clips were used, so results can vary a lot between splits.
- **No separate test set:** the best checkpoint was selected on the validation set and the final metrics were computed on that same set, so they may be slightly optimistic.
- **No data augmentation:** the model was trained on clips as they are, without noise, pitch shifts or time masking.
## 🔭 Next steps
 
- Hold out a **separate test set** for the final evaluation.
- Add **data augmentation** (background noise, SpecAugment).
- Train on the **full dataset** and compare against a pretrained audio model.
- Tune the decision threshold to trade precision for recall on the scream class.
## 🚀 How to run
 
1. Click the **Open in Colab** badge above (a GPU runtime is recommended).
2. Run all cells. The dataset downloads automatically with `kagglehub`.
To run it locally:
 
```bash
pip install torch torchaudio kagglehub numpy matplotlib tqdm
jupyter notebook CNN_AUDIO.ipynb
```
 
## 🛠️ Tech stack

Python · PyTorch · torchaudio · NumPy · Matplotlib · Google Colab
 
## 📚 References
 
- Human Screaming Detection Dataset, Kaggle.
- Huang, J.-T., Li, J., & Gong, Y. (2015). *An analysis of convolutional neural networks for speech recognition.* Microsoft Research.
- PyTorch-Adventures, *Intro to Audio Processing* notebook, by priyammaz (GitHub), used as a learning resource.
- OpenSLR (LibriSpeech), used for the audio sample in the introduction.
## 👤 Author
 
**Andrés Ramírez** · Systems and Computer Engineering student at Universidad Nacional de Colombia
[GitHub](https://github.com/Andres-cmk) · [LinkedIn](https://www.linkedin.com/in/andres-felipe-ramirez-m/)
 