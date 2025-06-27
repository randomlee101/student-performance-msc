## Student Performance Evaluation Version 2

This involves the use of 1-Dimensional Convolution Neural Network (CNN) typically used for numerical data analysis.

CNN was the deep learning technique of choice as a lot of input was considered for feature extraction as the input being considered is numerous and correlative analysis might not expose likely relationships between covariates

## 📊 Student Performance Data Preprocessing Script

This Python script cleans and preprocesses an augmented student performance dataset. It prepares the data for use in machine learning models or further statistical analysis by transforming categorical values, filtering irrelevant entries, and encoding features numerically.

---

## 📁 File Paths

- **Input:** `../assets/Performance_Augmented_3000.csv`
- **Output:** `../assets/Performance_Augmented_Cleaned.csv`

---

## ⚙️ Features & Transformations

### 1. **Load and Prepare Data**
- Reads the dataset using `pandas`.
- Drops the timestamp column.
- Removes duplicate rows (if any).

### 2. **Class Normalization**
- Converts class formats like `js2` or `SS3` to uniform representations like `JS 2`, `SS 3`.
- Filters out junior classes (`JS 1–3`) and keeps only senior classes (`SS 1–3`).

### 3. **After-School Activities**
- Splits multi-choice entries into individual activities using `.explode()`.
- Retains only popular activities occurring more than 1,000 times.

### 4. **Household Size Categorization**
- Groups raw values into defined ranges:
  - `1 - 3`
  - `4 - 6`
  - `7 - 10`

### 5. **Parental Qualification Standardization**
- Combines `"Masters Degree"` and `"PhD"` into a single label: `"Postgraduate"`.

### 6. **Grade and Position Binarization**
- Grades: Converts to two categories — `'64 & Below'` and `'65 & Above'`.
- Positions: Converts to `'Top 5'` and `'Top 10'`.

### 7. **Age Filtering**
- Keeps only ages that appear more than 2,000 times in the dataset.

### 8. **Label Encoding**
- Every column’s unique categorical values are encoded into numeric form using index-based mapping.

### 9. **Save Cleaned Data**
- The cleaned, normalized, and encoded dataset is saved to a new CSV file for modeling.

---

## 🧪 Example Columns Processed

- `Class`
- `Age`
- `Household Size`
- `After-School Activities`
- `Grade`
- `Position`
- `Qualification - Father`
- `Qualification - Mother`

---

## 💻 Requirements

```bash
pip install pandas numpy
```

## ⚖️ Grade Balancing Preprocessing Script

This script performs **class balancing** on the cleaned student performance dataset to ensure equal representation of grade categories (`64 & Below` and `65 & Above`) before training a machine learning model.

---

## 🧠 Objective

Machine learning models often perform poorly when the dataset is imbalanced. This script balances the dataset by:
- **Undersampling** the overrepresented class
- Ensuring both grade categories have an equal number of samples

---

## 📁 File Paths

- **Input File:** `../assets/Performance_Augmented_Cleaned.csv`  
- **Output File:** `../assets/Performance_Augmented_Preprocessed.csv`

---

## ⚙️ Key Steps

### 1. **Load the Cleaned Dataset**
```python
dataframe = pd.read_csv('../assets/Performance_Augmented_Cleaned.csv')
```

## 🤖 Student Grade Prediction using 1D Convolutional Neural Network (CNN)

This script trains a **1D CNN** model on a preprocessed student performance dataset to predict academic **Grade** categories (`64 & Below` vs `65 & Above`).

---

## 🧠 Model Objective

The goal is to **classify student grades** based on their socio-academic background using a deep learning model.

---

## 📁 File Paths

- **Input Dataset:** `../assets/Performance_Augmented_Preprocessed.csv`

---

## 📊 Data Overview

- The dataset is assumed to be **preprocessed and normalized**.
- The target variable is:
  - `Grade` (binary classification)
- The `Position` column is dropped and **not used** in this training process.

---

## ⚙️ Model Architecture

A `Sequential` model with the following layers:

| Layer Type          | Parameters                                   |
|---------------------|----------------------------------------------|
| `Conv1D`            | 16 filters, kernel size 2, ReLU activation   |
| `MaxPooling1D`      | Pool size 2, stride 1                        |
| `Flatten`           | Converts 3D tensor to 1D                     |
| `BatchNormalization`| Normalizes layer outputs                    |
| `Dense`             | 256 units, ReLU activation                  |
| `Dropout`           | 25% dropout rate to reduce overfitting      |
| `Dense`             | 1 unit, Sigmoid activation (for binary classification) |

---

## 🧪 Training Configuration

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Precision, Recall  
- **Validation Split:** 2%  
- **Epochs:** 60  
- **Batch Size:** 32  

---

## 📉 Training History Visualization

The training process is visualized using the external utility function:

```python
from data_visualization import plot_training_history
```