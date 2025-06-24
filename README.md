# Weight-Height Gender Classification using Random Forest  
This project performs gender classification based on a person's height and weight using a machine learning model. The dataset used includes synthetic height and weight data for males and females. The final model is trained using a Random Forest Classifier.

## Project Structure  
- Data_Mining_Project.ipynb       : Original Colab Notebook  
- gender_classification.py        : Clean Python version of the notebook  
- report.pdf                      : Project report document  
- README.md                       : This file  
- weight-height.csv               : Dataset used  

## Dataset  
Source: Weight-Height Dataset (https://www.kaggle.com/datasets/mustafaali96/weight-height)  
Columns:  
- Gender: Male/Female  
- Height: in inches  
- Weight: in pounds  

## Project Highlights  
- Data cleaning and preprocessing  
- Outlier detection and removal using IQR method  
- Outlier capping to reduce extreme values  
- Feature scaling using StandardScaler  
- Gender classification using RandomForestClassifier  
- Evaluation using accuracy, classification report, and confusion matrix  
- Visualization using seaborn/matplotlib  

## Model Performance  
- Final model used: Random Forest Classifier  
- Accuracy achieved: Approximately 97–98%  
- Evaluation includes: Confusion Matrix, Classification Report  

## How to Run  
1. Clone the repository:  
   git clone https://github.com/your-username/weight-height-gender-classification.git  
   cd weight-height-gender-classification  

2. Install dependencies (listed in requirements.txt):  
   pip install -r requirements.txt  

3. Run the notebook or script:  
   - Open Data_Mining_Project.ipynb in Google Colab or Jupyter Notebook  
   - Or run the script locally using:  
     python gender_classification.py  
