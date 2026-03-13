# NLP Project 1 – Hate Speech Classification Pipeline

**Course:** Introduction to NLP – FS 2026  
**Team:** 2 Personen  
**Deadline:** 18.03.2026  
**Grade:** 20% of final grade

---

## ✅ TODO

### Done  
- [x] "Baseline"-Code vom Dozenten verstanden (`hate_speech_classification.py`)  
- [x] `pipeline_functions.py` erstellt – modulare Funktionen für Preprocessing, Vectorizer, Modelle  
- [x] `project1_pipeline.ipynb` erstellt – Experimente einzeln ausführbar  
- [x] Duplicate Removal implementiert
- [x] Dataset exploriert (Label-Verteilung, Textlängen)  
- [x] E1 Baseline erfolgreich durchgelaufen
- [x] Testlauf Experimente mit `downsample=True` (zum Sicherstellen, dass der Code funktioniert)
- [x] E9 SVM Hyperparameter-Experiment implementiert (C=0.1, 1, 10)
- [x] Outlier Detection implementiert (sehr lange / sehr kurze Texte)
- [x] Finaler Durchlauf mit `downsample=False` für den Report  

### Todo  
- [ ] Learning curve für das beste Modell (hilfreich, nicht ein Muss)  
- [ ] Resultate analysieren und interpretieren  
- [ ] Evtl. weitere Tests falls nötig?
- [ ] Paper schreiben (ACL Format, max. 2 Seiten)  
- [ ] Paper abgeben bis 18.03.2026 12:00  

---  

## Goal

Build a systematic experimental pipeline for binary hate speech classification and compare how different **preprocessing**, **feature extraction**, and **model** choices affect performance.

---

## Project Structure

```
Project_1/
│
├── pipeline_functions.py        ← alle Funktionen (preprocessing, vectorizer, modelle, experiment runner,
│                                  outlier detection, hyperparameter tuning)
├── project1_pipeline.ipynb      ← Hauptnotebook: Experimente ausführen & Resultate visualisieren
├── hate_speech_classification.py← originaler Baseline-Code vom Dozenten (nicht verändern)
│
├── data/                        ← Kaggle-Daten (lokal, nicht in Git)
│   ├── train.csv
│   ├── test.csv
│   ├── test_labels.csv
│   └── sample_submission.csv
│
├── results.csv                  ← Experiment-Ergebnisse (wird automatisch generiert)
├── figures/                     ← Figures für das Paper (wird automatisch erstellt)
│   └── fig_*.png
│
├── .gitignore
└── README.md
```

> ⚠️ Der `data/` Ordner und alle `.pkl` / `.csv` Dateien sind in `.gitignore` – diese müssen lokal vorhanden sein (siehe unten).

> ⚠️ Der `figures/` Ordner wird automatisch erstellt (`os.makedirs('figures', exist_ok=True)`). `plt.savefig()` muss **vor** `plt.show()` aufgerufen werden, sonst wird ein leeres Bild gespeichert.

> ***ℹ️ Die wichtigsten Files für uns sind `pipeline_functions.py` und `project1_pipeline.ipynb`***

---

## Dataset herunterladen

1. Kaggle Account erstellen / einloggen
2. Dataset herunterladen: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
3. Folgende Files in den `data/` Ordner legen:
   - `train.csv` ← wird für alle Experimente verwendet
   - `test.csv`
   - `test_labels.csv`
   - `sample_submission.csv`

> ℹ️ `test.csv` und `test_labels.csv` könnten eigentlich für Testing benutzt werden (statt `train.csv` zu splitten), aber es gibt zu viele `-1` Values in `test_labels.csv`. Daher splitten wir `train.csv` in Train / Test im Code.

---

## Experimente ausführen

Einfach `project1_pipeline.ipynb` von oben nach unten ausführen. Das Notebook:

1. Lädt und analysiert den Datensatz
2. Führt alle Experimente automatisch durch
3. Speichert Resultate in `results.csv`
4. Generiert Figures in `figures/fig_*.png` für das Paper

**Beim ersten Ausführen** werden Cache-Files erstellt (`cache_*_X.pkl`, `cache_*_Y.pkl`) damit Preprocessing nicht jedes Mal wiederholt werden muss.

> ℹ️ E1–E6 sind momentan mit `downsample=True` (nimmt nur 20% der Daten → nur zum Testen ob der Code funktioniert). Für den Abschlussbericht `downsample=False` setzen, um alle Daten zu berücksichtigen (geht deutlich länger).

> ⚠️ Nach Änderungen an `pipeline_functions.py` immer den **Jupyter Kernel neu starten**, sonst werden veraltete Funktionsdefinitionen verwendet.

---

## Experiment-Übersicht

Prinzip: immer nur **eine Variable ändern**, die anderen auf Baseline-Wert fixieren.

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
| E9 | SVM Hyperparameter | Stem + Stopwords | TF-IDF (1000) | SVM (C=0.1 / 1 / 10) |

**Metric:** F1 Macro (da Datensatz unbalanciert: ~90% not hate speech / ~10% hate speech)  

*Wieso F1 Macro?*   
F1-Score ist das harmonische Mittel von Precision und Recall:  
`F1 = 2 * (Precision * Recall) / (Precision + Recall)`  
F1 Macro bedeutet: berechne den F1-Score separat für jede Klasse, dann nimm den Durchschnitt – ungewichtet, also jede Klasse zählt gleich viel.  
Bei uns: `F1 Macro = (F1 "not hate speech" + F1 "hate speech") / 2`

*Was ist mit Baseline gemeint?*  
Preprocessing: Tokenize + Stopword Removal + Stemming + Remove Numbers  
Features: TF-IDF (1000), Modell: SVM  
→ Entspricht dem `.py`-File vom Dozenten (`hate_speech_classification.py`)

### ℹ️ Warum diese Experimente?

| ID | Begründung |
|---|---|
| E1 | **Baseline** – Grundlage für alle Vergleiche. Ohne Baseline kann man keine Verbesserungen oder Verschlechterungen messen. |
| E2 | **Kein Preprocessing** – Wichtigste Kontrollfrage: Bringt das ganze Preprocessing überhaupt etwas? |
| E3 | **Stopwords, kein Stemming** – Isoliert den Effekt von Stemming. Vielleicht schadet Stemming sogar – bei Hate Speech könnten Wortformen wie "killed" vs "kill" relevant sein. |
| E4 | **Stemming, keine Stopwords** – Isoliert den Effekt von Stopword-Removal. Stopwords wie "you" oder "I" können bei aggressivem Text sogar informativ sein. |
| E5 | **Bag of Words statt TF-IDF** – TF-IDF gewichtet seltene Wörter höher, BoW behandelt alle gleich. Frage: Sind es eher häufige oder seltene Wörter, die Hate Speech signalisieren? |
| E6 | **TF-IDF mit 5000 Features** – Mehr Features = mehr Vokabular = eventuell bessere Abdeckung. Aber auch mehr Rauschen. Klassischer Tradeoff. |
| E7 | **Logistic Regression** – Schneller als SVM, oft ähnlich gut. Gewichte sind interpretierbar – man kann zeigen, welche Wörter am stärksten auf Hate Speech hinweisen. |
| E8 | **Naive Bayes** – Der klassische Text-Klassifikator, sehr schnell. Guter Vergleichspunkt zu SVM und Logistic Regression. |
| E9 | **SVM Hyperparameter (C=0.1 / 1 / 10)** – Wie stark beeinflusst der C-Parameter die Performance? C kontrolliert den Tradeoff zwischen Margin-Maximierung und Fehlertoleranz. |

---

## Outlier Detection

Die Pipeline enthält eine Outlier Detection für ungewöhnlich kurze oder lange Texte:

- **Sehr kurze Texte** (wenige Tokens): könnten zu wenig Information für die Klassifikation enthalten
- **Sehr lange Texte** (viele Tokens): könnten das Modell überproportional beeinflussen

Die Textlängen-Verteilung wird im Notebook visualisiert und als Figure in `figures/` gespeichert.

---

## Laufzeit

- Mit `downsample=True` (20% der Daten):
  - E1: ~7–8 min  
  - E2: ~7–8 min  
  - E3: ~5–6 min   
  - E4: ~10–11 min   
  - E5: ~1–2 min
  - E6: ~3–4 min
  - E7: < 1 min (bereits `downsample=False`)
  - E8: < 1 min (bereits `downsample=False`)
  - E9: ~7–8 min (3 SVM-Varianten)

- Mit `downsample=False` (alle Daten): SVM-Experimente (E1–E6, E9) dauern deutlich länger → für den finalen Report empfehlen wir, diese über Nacht laufen zu lassen.

---

## Paper

- Format: ACL Style, PDF, max. 2 Seiten + Figures/Tables + References
- Annotiert: wer hat welchen Teil geschrieben (author1 / author2 / joint)
- Figures werden automatisch in `figures/fig_*.png` gespeichert und können direkt ins Paper eingefügt werden
