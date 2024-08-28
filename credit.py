import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

file_path = "C:\\Users\\asvij\\Downloads\\veera\\Credit Score Classification Dataset.csv"
# Step 1: Load and prepare the dataset
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    X = data.drop('Credit Score', axis=1)
    y = data['Credit Score']
    
    return X, y, label_encoders
data.head()
# Step 2: Train the model and save it
def train_and_save_model(X, y, label_encoders, model_filename, preprocessing_filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, model_filename)
    preprocessing_objects = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'columns': X.columns  # Save the column order
    }
    joblib.dump(preprocessing_objects, preprocessing_filename)
# Step 3: Load the model and make predictions
def load_model_and_predict(new_data, model_filename, preprocessing_filename):
    # Load the saved model and preprocessing objects
    rf_model = joblib.load(model_filename)
    preprocessing_objects = joblib.load(preprocessing_filename)
    
    label_encoders = preprocessing_objects['label_encoders']
    scaler = preprocessing_objects['scaler']
    columns_order = preprocessing_objects['columns']
    for column, encoder in label_encoders.items():
        if column in new_data.columns:
            unseen_categories = set(new_data[column]) - set(encoder.classes_)
            if unseen_categories:
                raise ValueError(f"Unseen category in column '{column}': {unseen_categories}")
            new_data[column] = encoder.transform(new_data[column])
    new_data = new_data[columns_order]
    new_data_scaled = scaler.transform(new_data)
    predictions = rf_model.predict(new_data_scaled)
    predicted_class = label_encoders['Credit Score'].inverse_transform(predictions)
    return predicted_class
if __name__ == "__main__":
    model_filename = 'credit_score_model.pkl'
    preprocessing_filename = 'preprocessing.pkl'
    file_path = "C:\\Users\\asvij\\Downloads\\veera\\Credit Score Classification Dataset.csv"
    X, y, label_encoders = load_and_prepare_data(file_path)
    train_and_save_model(X, y, label_encoders, model_filename, preprocessing_filename)
    age = int(input("Age"))
    gender = input("Gender")
    income = int(input("Income"))
    education = input("Education")
    marital = input("Marital Status")
    child = int(input("No of childrens"))
    house = input("Home Ownership")
    new_sample = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Income': [income],
        'Education': [education],
        'Marital Status': [marital],
        'Number of Children': [child],
        'Home Ownership': [house]
    })
    
    try:
        predicted_score = load_model_and_predict(new_sample, model_filename, preprocessing_filename)
        print(f'Predicted Credit Score: {predicted_score[0]}')
    except ValueError as e:
        print(f"Error: {e}")
