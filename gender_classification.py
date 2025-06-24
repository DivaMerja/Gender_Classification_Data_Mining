# gender_classification.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('weight-height.csv')

# Encode Gender: Male = 0, Female = 1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Detect and remove outliers using IQR
Q1 = df[['Height', 'Weight']].quantile(0.25)
Q3 = df[['Height', 'Weight']].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Boolean mask for outliers
outliers = ((df[['Height', 'Weight']] < lower_bound) | (df[['Height', 'Weight']] > upper_bound)).any(axis=1)
df_cleaned = df[~outliers].copy()

# Apply capping to remaining data
df_cleaned['Height'] = df_cleaned['Height'].clip(lower=lower_bound['Height'], upper=upper_bound['Height'])
df_cleaned['Weight'] = df_cleaned['Weight'].clip(lower=lower_bound['Weight'], upper=upper_bound['Weight'])

# Features and target
X = df_cleaned[['Height', 'Weight']]
y = df_cleaned['Gender']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Boxplots (Before and After Outlier Handling)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Before
sns.boxplot(x='Gender', y='Height', data=df, ax=axes[0][0])
axes[0][0].set_title('Height Before Capping')

sns.boxplot(x='Gender', y='Weight', data=df, ax=axes[0][1])
axes[0][1].set_title('Weight Before Capping')

# After
sns.boxplot(x='Gender', y='Height', data=df_cleaned, ax=axes[1][0])
axes[1][0].set_title('Height After Capping')

sns.boxplot(x='Gender', y='Weight', data=df_cleaned, ax=axes[1][1])
axes[1][1].set_title('Weight After Capping')

plt.tight_layout()
plt.show()
