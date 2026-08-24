# 📝 Notes Actionizer

Notes Actionizer is an NLP-powered web application that converts messy class, meeting, or project notes into structured information and actionable to-do items.

Built using **Python, spaCy, Gradio, and Pandas**.

## ✨ Features

- 🔤 Tokenization
- 🏷️ Part-of-Speech (POS) Tagging
- 🔄 Lemmatization
- 🧹 Stopword & Punctuation Removal
- 🌍 Named Entity Recognition (NER)
- ✅ Automatic Action Item Extraction
- 📊 Structured NLP Results
- ⚡ Interactive Gradio Interface

## 🧠 NLP Pipeline

```text
Raw Notes
   ↓
Tokenization
   ↓
POS Tagging
   ↓
Lemmatization
   ↓
Stopword Removal
   ↓
Named Entity Recognition
   ↓
Action Item Extraction
   ↓
Clean To-Do List

notes-actionizer/
│
├── app.py
├── Procfile
├── requirements.txt
└── README.md
git clone https://github.com/AyushiiDhiman/notes-actionizer.git
cd notes-actionizer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
http://localhost:7860
