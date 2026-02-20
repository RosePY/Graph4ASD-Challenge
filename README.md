# 🧠 Graph4ASD Challenge 

**[Live Leaderboard](https://rosepy.github.io/Graph4ASD-Challenge)**

## 📌 Overview

This repository hosts the **Graph4ASD Challenge**, a graph machine learning competition focused on **Autism Spectrum Disorder (ASD) classification** using resting-state fMRI functional connectivity data from the ABIDE dataset.

Participants must design and train **Graph Neural Network (GNN)** models to classify subjects as:

- ASD  
- Typical Control (TC)  

based on brain connectivity graphs.

Each subject is represented as a graph where:

- Nodes → brain regions (Craddock 200 atlas)  
- Edges → functional connectivity between brain regions  

---

## 🏆 Leaderboard

Leaderboard scores are automatically updated.

👉 **[Live Leaderboard](https://rosepy.github.io/Graph4ASD-Challenge)**

---

## 🧠 Task Description

Each sample corresponds to one subject’s brain graph.

### Graph Definition

\[
G = (A, X)
\]

Where:

- **A** → adjacency matrix (functional connectivity)  
- **X** → node feature matrix (identity features)
- **Nodes** → 200 brain regions (Craddock atlas)  

### Objective

Train a model that correctly predicts whether a subject:

- has Autism Spectrum Disorder (ASD)  
- or is a Typical Control (TC)  

using:

- Graph structure  
- Node features  

This is a **graph-level classification task**.

---

## 📊 Dataset Details

The dataset is derived from:

> **Autism Brain Imaging Data Exchange (ABIDE)**

ABIDE aggregates anonymized rs-fMRI data from 17 international sites.

### Full Dataset (original ABIDE)

| Property | Value |
|----------|-------|
| Subjects | 1,009 |
| ASD | 516 (51.14%) |
| Controls | 493 (48.86%) |
| Brain atlas | Craddock 200 |

⚠️ For the challenge, a processed subset is provided, 484 samples for train and 153 for test. Participants must use data as-is, since original data has been exclusively processed for this challenge. 

---

## 🧩 Data Representation

Each subject is stored as a graph:

- **Adjacency matrix (A)** → connectivity between brain regions  
- **Feature matrix (X)** → node-level features  

All graphs share:

- Same number of nodes → 200  
- Same node ordering → Craddock atlas  

---

## 📂 Dataset Access

The dataset is hosted in folder `data/public`. 



## 📝 How to Submit Your Results

### Step 1: Train
Train your model using:
- Graphs: Adjacency matrix and node feature matrix in `data/public/adj_train.npy` and `data/public/node_features.npy`. Each 
- labels in `data/public/train_label.csv`

### Step 2: Predict
Predict labels for every graph in:
- `data/public/adj_test.npy` and `data/public/node_features_test.npy`

### Step 3: Prepare your submission file
Create a CSV with columns `filename` and `prediction` (same format as `submissions/sample_submission.csv`):

```csv
id,label
1,0
2,0
3,1
...
```
Save it as a `.csv` file (e.g. `my_submission.csv`) in the **`submissions/`** folder.  
**Note:** ❗ Do NOT upload the raw CSV. You need to submit an **encrypted** version of your predictions file to keep privacy.

### Step 4: Encrypt your submission
From the project root, run the encryption script so it can find your CSV and the encryption key:

```bash
cd submissions
python ../encryption/encrypt_submissions.py <filename>
cd ..
```

This creates a `.enc` file next to each `.csv` in `submissions/` (e.g. `my_submission.csv.enc`). Only `.enc` files are tracked by git; your `.csv` stays local.

### Step 5: Create a Pull Request
Commit and push the new `.enc` file(s) to the repository (e.g. open a Pull Request or push to the main branch, as per the challenge rules). The automated pipeline will decrypt and score your submission to update the leaderboard.

⚠️ Only **one submission per participant** is allowed.

## 🏁 Challenge Rules

- You are free to use any Graph Neural Network.
- You are not allowed to use external data.
- You must not try to identify each subject.


## 📚 References

Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden.

