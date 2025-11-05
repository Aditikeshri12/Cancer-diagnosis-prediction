# 🧠 Predictive Modeling on Breast Cancer Dataset  

## 📌 Overview  
This project focuses on building and comparing multiple **predictive machine learning models** to classify breast cancer as **malignant (1)** or **benign (0)** using the **Wisconsin Breast Cancer Dataset** (`wisc_bc_data.csv`).  

The workflow includes **data preprocessing**, **model training**, **evaluation**, and **visualization** to identify the best-performing predictive model.  

Both **R** and **Python** implementations are included in this repository.

---

## 📂 Files in Repository  
- `Predictive_Project.R` → R script for predictive modeling  
- `Predictive_Project.py` → Python script for predictive modeling  
- `wisc_bc_data.csv` → Dataset  
- `README.md` → Project documentation  

---

## ⚙️ Installation and Setup  

### ✅ R Setup  

#### Install Required Libraries  
```R
install.packages(c("e1071", "class", "rpart", "ggplot2", "caTools", "caret", "dplyr", "tidyr"))
```

#### Load Libraries  
```R
library(caret)
library(class)
library(e1071)
library(rpart)
library(ggplot2)
library(dplyr)
library(tidyr)
```

### ✅ Python Setup  

#### Install Required Python Packages  
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

#### Import Libraries in Python Script  
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 📂 Dataset Loading  

### R  
```R
data <- read.csv("wisc_bc_data.csv")
```

### Python  
```python
data = pd.read_csv("wisc_bc_data.csv")
```

---

## 🔄 Data Preprocessing  
- Convert diagnosis column:
  - `M` → `1` (Malignant)  
  - `B` → `0` (Benign)  
- Remove non-essential columns (like `id`)  
- Train-test split (70% / 30%)  
- Standardize features  

---

## 🤖 Models Used  
| Model | Purpose |
|-------|--------|
| KNN | Distance-based classification |
| Naive Bayes | Probabilistic classifier |
| Decision Tree | Tree-based classification |
| Logistic Regression | Binary predictive model |
| K-Means | Unsupervised model used for comparison | 

---

## 📊 Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Error Rate  

Visualization done using **ggplot2** (R) and **matplotlib / seaborn** (Python).

---

## 🚀 How to Run  

### R  
1. Open `Predictive_Project.R`
2. Run line by line in RStudio  

### Python  
```bash
python Predictive_Project.py
```

---

## 🧩 Future Enhancements  
- Add SVM & Random Forest  
- Use Cross-validation  
- Add ROC & AUC metrics  

---

## 👩‍💻 Author  
**Aditi Kumari**  
B.Tech (CSE) — Lovely Professional University  
📍 Bhagalpur, Bihar, India  

---

## 🏷️ Tags  
`Machine Learning` • `Predictive Analytics` • `Breast Cancer Dataset` • `Python ML` • `R ML` • `Healthcare AI`
