# Weight-Height Gender Classification using Random Forest

This project performs gender classification based on a person's height and weight using a machine learning model. The dataset used includes synthetic height and weight data for males and females. The final model is trained using a Random Forest Classifier.

## 📁 Project Structure


## 📊 Dataset

- **Source:** [Weight-Height Dataset](https://www.kaggle.com/datasets/mustafaali96/weight-height)
- **Columns:**
  - `Gender`: Male/Female
  - `Height`: in inches
  - `Weight`: in pounds

## 📌 Project Highlights

- Data cleaning and preprocessing
- Outlier detection and removal using IQR method
- Outlier capping to reduce extreme values
- Feature scaling using `StandardScaler`
- Gender classification using `RandomForestClassifier`
- Evaluation using accuracy, classification report, and confusion matrix
- Visualization using seaborn/matplotlib

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/weight-height-gender-classification.git
   cd weight-height-gender-classification
