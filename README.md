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
- [x] `project1_pipeline.ipynb` erstellt – 8 Experimente einzeln ausführbar  
- [x] Duplicate Removal implementiert
- [x] Dataset exploriert (Label-Verteilung, Textlängen)  
- [x] E1 Baseline erfolgreich durchgelaufen
- [x] Testlauf Experimente mit `downsample=True` (zum sicherstellen das der Code funktioniert)

### Todo  
- [X] Ein einfaches Hyperparameter-Experiment (z.B. SVM mit C=0.1, 1, 10)
- [X] Outlier detection (sehr lange / sehr kurze Texte)
- [ ] Learning curve für das beste Modell (hilfreich, nicht ein muss i guess)
- [ ] Finaler Durchlauf mit `downsample=False` für den Report  
- [ ] Resultate analysieren und interpretieren  
- [ ] Evtl. wietere Tests falls nötig ?
- [ ] Bilder die erstellt werdne in seperaten Ordner speichern (falls die überhaupt gebraucht werden)
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
├── pipeline_functions.py        ← alle Funktionen (preprocessing, vectorizer, modelle, experiment runner)
├── project1_pipeline.ipynb      ← Hauptnotebook: Experimente ausführen & Resultate visualisieren
├── hate_speech_classification.py← originaler Baseline-Code vom Dozenten (nicht verändern)
│
├── results.csv                  ← Experiment-Ergebnisse (wird automatisch generiert)
├── fig_*.png                    ← Figures für das Paper (werden automatisch generiert)
│
├── .gitignore
└── README.md
```

> ⚠️ Die `/data` Ordner und alle `.pkl` / `.csv` Dateien sind in `.gitignore` – diese müssen lokal vorhanden sein (siehe unten).

> ***ℹ️ Die Wichtigsten Files für uns ist pipeline_functions.py und project1_pipeline.ipynb***


---

## Dataset herunterladen

1. Kaggle Account erstellen / einloggen
2. Dataset herunterladen: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
3. Folgende Files in den `data` Ordner legen:
   - `train.csv` ← wird für alle Experimente verwendet
   - `test.csv`
   - `test_labels.csv`
   - `sample_submission.csv`

> ℹ️test.csv und test_labels, könnte eigentlich für testing benutzt werden (statt train.csv zu spliten) aber gibt zu viele -1 values in test_labels.csv
daher spliitn wir den train.csv in train / test im Code.  


---

## Experimente ausführen

Einfach `project1_pipeline.ipynb` von oben nach unten ausführen. Das Notebook:

1. Lädt und analysiert den Datensatz
2. Führt alle 8 Experimente automatisch durch
3. Speichert Resultate in `results.csv`
4. Generiert Figures (`fig_*.png`) für das Paper

**Beim ersten Ausführen** werden Cache-Files erstellt (`cache_*_X.pkl`, `cache_*_Y.pkl`) damit preprocessing nicht jedes Mal wiederholt werden muss.

> ℹ️ E1 - E6 sind momentan downsampling = True (nimmt nur 20% der Daten --> geht nur drum zum gucken, ob es wirklich funktioniert) für den Abschluss Bericht amcht es Sinn downsample = False zu setzten, um alle daten zu berücksichtigen, geht einfach sehr lange)


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

*Wieso F1 Macro?*   
F1-Score ist das harmonische Mittel von Precision und Recall:  
F1 = 2 * (Precision * Recall) / (Precision + Recall)  
F1 Macro bedeutet: berechne den F1-Score separat für jede Klasse, dann nimm den Durchschnitt – ungewichtet, also jede Klasse zählt gleich viel.  
Bei euch: F1 Macro = (F1 "not toxic" + F1 "toxic") / 2   
   
*Was ist mit Basline gemeint*  
Preprocessing: Tokenize + Stopword removal + Stemming + Remove numbers   
Features: TF-IDF (1000) und Model: SVM  
wird bereits im .py file vom Dozent gemacht (hate_speech_classification.py) --> Claude hat es baseline genannt  


### ℹ️ Warum diese 8 Experimente?

| ID | Begründung |
|---|---|
| E1 | **Baseline** – Grundlage für alle Vergleiche. Ohne Baseline kann man keine Verbesserungen oder Verschlechterungen messen. |
| E2 | **Kein Preprocessing** – Wichtigste Kontrollfrage: Bringt das ganze Preprocessing überhaupt etwas? Wenn E2 ähnlich gut ist wie E1, war der Aufwand umsonst. |
| E3 | **Stopwords, kein Stemming** – Isoliert den Effekt von Stemming. Vielleicht schadet Stemming sogar – bei Hate Speech könnten Wortformen wie "killed" vs "kill" relevant sein. |
| E4 | **Stemming, keine Stopwords** – Isoliert den Effekt von Stopword-Removal. Stopwords wie "you" oder "I" können bei aggressivem Text sogar informativ sein. |
| E5 | **Bag of Words statt TF-IDF** – TF-IDF gewichtet seltene Wörter höher, BoW behandelt alle gleich. Frage: Sind es eher häufige oder seltene Wörter die Hate Speech signalisieren? |
| E6 | **TF-IDF mit 5000 Features** – Mehr Features = mehr Vokabular = eventuell bessere Abdeckung. Aber auch mehr Rauschen. Klassischer Tradeoff. |
| E7 | **Logistic Regression** – Schneller als SVM, oft ähnlich gut. Die Gewichte sind interpretierbar – man kann zeigen welche Wörter am stärksten auf Hate Speech hinweisen. |
| E8 | **Naive Bayes** – Der klassische Text-Klassifikator, sehr schnell. Guter Vergleichspunkt zu SVM und Logistic Regression. |


---

## Laufzeit

- Mit `downsample=True` (20% der Daten):
  E1: ~ 7-8 min  
  E2: ~ 7-8 min  
  E3: ~ 5-6 min   
  E4: ~ 10-11 min   
  E5: ~ 1-2 min
  E6: ~ 3-4 min
  E7: ~ < 1min (bereits downsample=False)
  E8: ~  < 1min (bereits downsample=False)

- Mit `downsample=False` (alle Daten): SVM-Experimente können einges länder dauern für E1-E6 → für finalen Report über Nacht laufen lassen (empfehlung von Claude)

---

## Paper

- Format: ACL Style, PDF, max. 2 Seiten + Figures/Tables + References
- Annotiert: wer hat welchen Teil geschrieben (author1 / author2 / joint)
- Figures werden automatisch als `fig_*.png` gespeichert und können direkt ins Paper eingefügt werden
