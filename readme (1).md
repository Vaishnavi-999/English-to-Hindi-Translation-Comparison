# 🌐 English–Hindi Translation Projects (NLP)

This repository showcases **two end-to-end NLP translation projects** built using the **IIT Bombay English–Hindi dataset** from Hugging Face. The projects cover the **complete lifecycle** of an NLP system — from **model building and fine-tuning** to **API deployment** using **FastAPI** and **Hugging Face Spaces**.

---

## 🚀 Projects Overview

| Project | Model Type | Description | Live API |
|-------|-----------|-------------|---------|
| **Project 1** | LSTM + Encoder–Decoder | Classic Seq2Seq neural machine translation | ✅ Available |
| **Project 2** | Fine-tuned Transformer | Improved translation using fine-tuning & data polishing | ✅ Available |

---

## 📌 Project 1: English → Hindi Translation (LSTM Encoder–Decoder)

### 🔹 Description
This project implements a **Neural Machine Translation (NMT)** system using:
- **LSTM-based Encoder–Decoder architecture**
- **Sequence-to-Sequence learning**
- **Teacher Forcing** during training

⚠️ **Important Note on Performance**  
This model is **trained from scratch** on the IITB dataset without using any pre-trained language knowledge. As a result:
- The model **does not provide very accurate or fluent translations**
- Accuracy is **relatively low** compared to modern approaches
- Translations may sound **grammatically incorrect or incomplete**

👉 This limitation is **intentional and educational**, as the goal of Project 1 is to understand **core NLP and Seq2Seq fundamentals**, not to achieve production-level accuracy.

### 🔹 Dataset
- **IIT Bombay English–Hindi Parallel Corpus**
- Source: Hugging Face
- Used for supervised sequence-to-sequence learning

### 🔹 Architecture
```
English Sentence
      ↓
 Tokenization
      ↓
 LSTM Encoder → Context Vector → LSTM Decoder
      ↓
 Hindi Sentence
```

### 🔹 Key Features
- Text preprocessing (cleaning, tokenization, padding)
- Separate encoder and decoder models
- Separate training and inference logic
- Demonstrates limitations of training from scratch

### 🔹 Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pickle
- FastAPI
- Hugging Face Spaces

### 🔹 Live API
🔗 **English → Hindi Translation API (Baseline Model)**  
👉 https://huggingface.co/spaces/patil1203/english_to_hindi_translation

---


## 📌 Project 2: Fine-Tuned English–Hindi Translator (Improved Model)

### 🔹 Description
This project is an **enhanced and production-ready version** of Project 1. Instead of training from scratch, this model uses **fine-tuning on a pre-trained translation model**, combined with **dataset polishing and optimization**.

✅ **Key Result**  
- Fine-tuning enables the model to leverage **pre-learned language representations**
- Translation quality improves **significantly**
- Achieves **~95%+ translation accuracy** (approx.) on validation data
- Outputs are **more fluent, grammatically correct, and context-aware**

This clearly demonstrates **why fine-tuning outperforms training from scratch** in real-world NLP applications.

### 🔹 Improvements Over Project 1
- Uses pre-trained transformer-based model
- Fine-tuned on IITB English–Hindi dataset
- Cleaner and polished training data
- Much higher accuracy and fluency
- Faster convergence and better generalization

### 🔹 Workflow
```
Dataset Polishing
      ↓
 Tokenization & Preprocessing
      ↓
 Fine-Tuning Pre-trained Model
      ↓
 Evaluation (≈95%+ Accuracy)
      ↓
 API Deployment
```

### 🔹 Tech Stack
- Python
- Transformers (Hugging Face)
- PyTorch / TensorFlow
- FastAPI
- Hugging Face Spaces

### 🔹 Live API
🔗 **Fine-Tuned English–Hindi Translation API (High Accuracy)**  
👉 https://patil1203-en-hi-translator-api.hf.space/

---


## 🧠 Key NLP Concepts Demonstrated
- Sequence-to-Sequence Learning
- Encoder–Decoder Architecture
- LSTM Networks
- Tokenization & Padding
- Fine-Tuning Pre-trained Models
- Model Deployment as REST API
- End-to-End ML Product Development

---

## 📂 Repository Structure (Suggested)
```
├── project-1-lstm-encoder-decoder/
│   ├── training_notebook.ipynb
│   ├── app.py
│   ├── models/
│   └── tokenizer/
│
├── project-2-fine-tuned-model/
│   ├── fine_tuning_notebook.ipynb
│   ├── app.py
│   └── model/
│
└── README.md
```

---

## 🎯 Learning Outcomes
- Built NMT models from scratch
- Understood limitations of LSTM-based translation
- Applied fine-tuning to improve NLP performance
- Deployed ML models as real-world APIs
- Used Hugging Face ecosystem effectively

---

## 🙌 Acknowledgements
- **IIT Bombay** for the English–Hindi dataset
- **Hugging Face** for datasets, transformers & Spaces

---

## 📬 Contact
If you have suggestions or want to collaborate:
- 💼 LinkedIn: *Add your LinkedIn here*
- 📧 Email: *Add your email here*

---

⭐ If you like this project, don’t forget to star the repository!

