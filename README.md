# 🧠 Predictive Modeling on Breast Cancer Dataset  

## 📌 Overview  
This project focuses on building and comparing multiple **predictive machine learning models** to classify breast cancer as **malignant (1)** or **benign (0)** using the **Wisconsin Breast Cancer Dataset** (`wisc_bc_data.csv`).  

The workflow includes **data preprocessing**, **model training**, **evaluation**, and **visualization** to identify the best-performing predictive model.  

---

## 📂 Files in Repository  
- `Predictive_Project.R` → R script with full implementation of predictive modeling, evaluation, and visualization.  
- `wisc_bc_data.csv` → Dataset used for model training and testing.  
- `README.md` → Project documentation (this file).  

---

## ⚙️ Installation and Setup  

### 1️⃣ Install Required Libraries  
Run the following command to install necessary packages:  
```R
install.packages(c("e1071", "class", "rpart", "ggplot2", "caTools", "caret", "dplyr", "tidyr"))
```

### 2️⃣ Load the Libraries  
```R
library(caret)
library(class)
library(e1071)
library(rpart)
library(ggplot2)
library(dplyr)
library(tidyr)
```

### 3️⃣ Load the Dataset  
Select the dataset when prompted:  
```R
data <- read.csv(file.choose())
```
Or load directly if in the working directory:  
```R
data <- read.csv("wisc_bc_data.csv")
```

---

## 🔄 Data Preprocessing  
- Converted **diagnosis (M/B)** to binary format:  
  - `M` → `1` (Malignant)  
  - `B` → `0` (Benign)  
- Removed irrelevant columns (`id`).  
- Split dataset into **70% training** and **30% testing**.  
- Standardized features using `caret::preProcess` for model consistency.  

---

## 🤖 Models Used for Prediction  
| Model | Description |
|--------|--------------|
| **K-Nearest Neighbors (KNN)** | Distance-based algorithm; works well with standardized data. |
| **Naive Bayes** | Probabilistic model based on Bayes’ theorem; assumes feature independence. |
| **Decision Tree** | Handles non-linear relationships effectively but may overfit. |
| **Logistic Regression** | Suitable for binary prediction with linear boundaries. |
| **K-Means Clustering** | Unsupervised model used for comparative purposes. |

---

## 📊 Evaluation Metrics  
Each predictive model is evaluated using:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Error Rate**  

Custom metric calculation function ensures consistency across models.  

---

## 🎨 Visualization  
Performance comparison is shown using bar plots generated via `ggplot2`:  

```R
ggplot(results_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme_minimal() +
  ggtitle("Predictive Model Comparison Across Evaluation Metrics") +
  ylab("Metric Value") +
  xlab("Evaluation Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

---

## 🧾 Results Summary  
All predictive models’ performance metrics are combined into a data frame for visualization.  
This enables easy comparison and selection of the most accurate prediction model for breast cancer diagnosis.  

---

## 💡 Key Insights  
- **KNN** and **Logistic Regression** perform well on clean, standardized data.  
- **Decision Tree** captures non-linearity but may require pruning.  
- **Naive Bayes** performs well with independent features.  
- **K-Means** serves as an exploratory reference model.  

---

## 🚀 How to Run  
1. Open `Predictive_Project.R` in **RStudio**.  
2. Run the script step-by-step (Ctrl + Enter).  
3. Upload `wisc_bc_data.csv` when prompted.  
4. Check console and plot window for model outputs and comparison charts.  

---

## 🧩 Future Enhancements  
- Add **Support Vector Machine (SVM)** and **Random Forest** for improved predictive power.  
- Implement **cross-validation** to prevent overfitting.  
- Include **ROC Curve** and **AUC** analysis for better evaluation.  

---

## 👩‍💻 Author  
**Aditi Kumari**  
B.Tech (CSE) — Lovely Professional University  
📍 Bhagalpur, Bihar, India  

---

## 🏷️ Keywords  
`Predictive Modeling` • `Machine Learning` • `Breast Cancer` • `Classification` • `R Programming` • `Data Science`
