import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv(r'D:\SEM 7\SEM 7\AI-ML(Honars)\project\drugs_side_effects_drugs_com.csv')

# Select features and target
X = data[['drug_name', 'side_effects', 'generic_name', 'drug_classes', 'brand_names', 'activity', 'rx_otc', 'pregnancy_category', 'csa', 'alcohol']]
y = data['medical_condition']

# Encoding categorical variables
X_encoded = pd.get_dummies(X)
y_encoded = y  # Use original labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'model.pkl')
print("Model trained and saved as model.pkl")
