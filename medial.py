import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv("C:\\Users\\asvij\\Downloads\\veera\\archive\\diabetes.csv")
print(data)
columns_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zero_values] = data[columns_with_zero_values].replace(0, np.nan)

data.fillna(data.median(), inplace=True)
X = data.drop('Outcome', axis=1)  
y = data['Outcome']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
import joblib
joblib.dump(model, 'diabetes_prediction_model.pkl')


def get_user_input():
    print("Enter the following medical data:")
    pregnancies = float(input("Number of Pregnancies: "))
    glucose = float(input("Glucose level: "))
    blood_pressure = float(input("Blood Pressure level: "))
    skin_thickness = float(input("Skin Thickness (mm): "))
    insulin = float(input("Insulin level: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
    
    
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    return user_data


user_input = get_user_input()


prediction = model.predict(user_input)


if prediction[0] == 1:
    print("The model predicts that the individual is likely to have diabetes.")
else:
    print("The model predicts that the individual is unlikely to have diabetes.")