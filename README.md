# NLP Project 1 – Hate Speech Classification Pipeline

**Course:** Introduction to NLP – FS 2026  
**Team:** 2 Personen  
**Deadline:** 18.03.2026  
**Grade:** 20% of final grade

---

## Goal

Build a systematic experimental pipeline for binary hate speech classification and compare how different **preprocessing**, **feature extraction**, and **model** choices affect performance.

---

## Project Structure

```
Project_1/
│
├── pipeline.py                  ← alle Funktionen (preprocessing, vectorizer, modelle, experiment runner)
├── pipeline.ipynb               ← Hauptnotebook: Experimente ausführen & Resultate visualisieren
├── hate_speech_classification.py← originaler Baseline-Code vom Dozenten (nicht verändern)
│
├── results.csv                  ← Experiment-Ergebnisse (wird automatisch generiert)
├── fig_*.png                    ← Figures für das Paper (werden automatisch generiert)
│
├── .gitignore
└── README.md
```

> ⚠️ Die `/data` Ordner und alle `.pkl` / `.csv` Dateien sind in `.gitignore` – diese müssen lokal vorhanden sein (siehe unten).

---

## Dataset herunterladen

1. Kaggle Account erstellen / einloggen
2. Dataset herunterladen: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
3. Folgende Files in den `Project_1` Ordner legen (gleiche Ebene wie `pipeline.py`):
   - `train.csv` ← wird für alle Experimente verwendet
   - `test.csv`
   - `test_labels.csv`
   - `sample_submission.csv`

---

## Experimente ausführen

Einfach `pipeline.ipynb` von oben nach unten ausführen. Das Notebook:

1. Lädt und analysiert den Datensatz
2. Führt alle 8 Experimente automatisch durch
3. Speichert Resultate in `results.csv`
4. Generiert Figures (`fig_*.png`) für das Paper

**Beim ersten Ausführen** werden Cache-Files erstellt (`cache_*_X.pkl`, `cache_*_Y.pkl`) damit preprocessing nicht jedes Mal wiederholt werden muss.

E1 - E6 sind momentan downsampling = True (nimmt nur 20% der Daten --> geht nur drum zum gucken, ob es wirklich funktioniert) für den Abschluss Bericht amcht es Sinn downsample = False zu setzten, um alle daten zu berücksichtigen, geht einfach sehr lange)
---

## Experiment-Übersicht

Prinzip: immer nur **eine Variable ändern**, die anderen zwei auf Baseline-Wert fixieren.

| ID | Name | Preprocessing | Features | Modell |
|---|---|---|---|---|
| E1 | Baseline | Stem + Stopwords | TF-IDF (1000) | SVM |
| E2 | No Preprocessing | – | TF-IDF (1000) | SVM |
| E3 | Stopwords only | Stopwords | TF-IDF (1000) | SVM |
| E4 | Stemming only | Stem | TF-IDF (1000) | SVM |
| E5 | Bag of Words | Stem + Stopwords | BoW (1000) | SVM |
| E6 | TF-IDF 5000 | Stem + Stopwords | TF-IDF (5000) | SVM |
| E7 | Logistic Regression | Stem + Stopwords | TF-IDF (1000) | LogReg |
| E8 | Naive Bayes | Stem + Stopwords | TF-IDF (1000) | NaiveBayes |

**Metric:** F1 Macro (da Datensatz unbalanciert: 90% not toxic / 10% toxic)

**Wieso F1 Macro?**   
F1-Score ist das harmonische Mittel von Precision und Recall:  
F1 = 2 * (Precision * Recall) / (Precision + Recall)  
F1 Macro bedeutet: berechne den F1-Score separat für jede Klasse, dann nimm den Durchschnitt – ungewichtet, also jede Klasse zählt gleich viel.  
Bei euch: F1 Macro = (F1 "not toxic" + F1 "toxic") / 2  
  

**Was ist mit Basline gemeint**  
Preprocessing: Tokenize + Stopword removal + Stemming + Remove numbers  
Features: TF-IDF (1000) und Model: SVM 
wird bereits im .py file vom Dozent gemacht (hate_speech_classification.py) --> Claude hat es baseline genannt
---

## Laufzeit

- Mit `downsample=True` (20% der Daten):
  E1: ~ 6 - 7min
  E2:  
  E3:  
  E4:  
  E5:  
- Mit `downsample=False` (alle Daten): SVM-Experimente können 30–60 Min dauern → für finalen Report über Nacht laufen lassen

---

## Paper

- Format: ACL Style, PDF, max. 2 Seiten + Figures/Tables + References
- Annotiert: wer hat welchen Teil geschrieben (author1 / author2 / joint)
- Figures werden automatisch als `fig_*.png` gespeichert und können direkt ins Paper eingefügt werden