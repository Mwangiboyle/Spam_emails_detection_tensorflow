# 📧 Spam Email Detection with GRU and BERT

A complete end-to-end NLP pipeline that detects spam emails using deep learning models. This project includes training with GRU and fine-tuning a BERT model, evaluating performance, saving models, and setting up for deployment.

---

## 🚀 Project Overview

- 🧹 Text Preprocessing & Tokenization
- 🧠 Model 1: GRU (RNN-based model in TensorFlow/Keras)
- 🧠 Model 2: BERT fine-tuning (HuggingFace Transformers)
- 📊 Evaluation using Accuracy, Precision, Recall, F1-Score
- 💾 Model Saving for deployment (`.keras` and `transformers` format)
- ⚙️ (Optional) FastAPI or Streamlit endpoint for predictions

---

## 📂 Project Structure

```
.
├── data/
│   └── spam.csv
├── models/
│   ├── gru_spam_model/
│   └── bert_spam_model/
├── notebooks/
│   └── eda_and_experiments.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_gru.py
│   ├── train_bert.py
│   └── evaluate.py
├── app/
│   └── predict_api.py
├── Dockerfile (optional)
├── requirements.txt
└── README.md
```

---

## 🧠 Models

### 1. GRU Model (Keras)
- Embedding + GRU + Dropout + Dense
- Tokenized using Keras `Tokenizer`
- Saved using `.h5` or TensorFlow SavedModel format

### 2. BERT Model (Transformers)
- Pretrained `bert-base-uncased` model
- Fine-tuned for binary classification (spam vs not spam)
- Saved using `model.save_pretrained()` and `tokenizer.save_pretrained()`

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| GRU   | ~96.16%     | 93.4%       | 90.1%    | 94.23%      |
| BERT  | ~73.45%     | 82.6%       | 78.2%    | 77.67%      |


---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Mwangiboyle/Spam_emails_detection_tensorflow.git
cd Spam_emails_detection_tensorflow

# Setup environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💾 Save/Load Models

```python
# GRU Model
model.save("models/gru_spam_model.keras")

# BERT Model
model.save_pretrained("models/bert_spam_model")
tokenizer.save_pretrained("models/bert_spam_model")
```

---

## 🧪 How to Train

```bash
# Train GRU
python src/train_gru.py

# Train BERT
python src/train_bert.py
```

---

## 🌐 Predict API (Optional)

```bash
# Run the FastAPI or Streamlit app
uvicorn app.predict_api:app --reload
```

---

## 📦 Git LFS (if using large model files)

```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.bin"
```

---

## 🛡️ License

MIT License © 2025 [Your Name]

---

## 🤝 Contributions

Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change.

---

## 📬 Contact

Reach out via [LinkedIn](https://www.linkedin.com/in/josephmwangiboyle/) or email: mwangiboyle4@gmail.com
