# 📧 Spam Email Detection with GRU and BERT

A complete end-to-end NLP pipeline that detects spam emails using deep learning models. This project includes training with GRU and fine-tuning a BERT model, evaluating performance, saving models, and setting up for deployment.

---

## 🚀 Project Overview

- 🧹 Text Preprocessing & Tokenization
- 🧠 Model 1: GRU (RNN-based model in TensorFlow/Keras)
- 🧠 Model 2: BERT fine-tuning (HuggingFace Transformers)
- 📊 Evaluation using Accuracy, Precision, Recall, F1-Score
- 💾 Model Saving for deployment (`.h5` and `transformers` format)
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
| GRU   | ~XX%     | XX%       | XX%    | XX%      |
| BERT  | ~XX%     | XX%       | XX%    | XX%      |

> Replace `XX%` with your actual results after training.

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/spam-detector-nlp.git
cd spam-detector-nlp

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
model.save("models/gru_spam_model")

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

Reach out via [LinkedIn](https://www.linkedin.com/) or email: your.email@example.com
