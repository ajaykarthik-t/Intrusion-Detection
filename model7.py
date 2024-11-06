import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from sklearn.svm import SVC
import joblib
import numpy as np
import threading

# Load the trained models and pre-processing tools
cnn_model = keras.models.load_model('nn_model.h5')
svm_model = joblib.load('svm_model.sav')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize global variables
test_data = None
cnn_results = []
svm_results = []

# Create a function to preprocess the test data and perform classification
def classify_test_data():
    global cnn_results, svm_results
    cnn_results = []
    svm_results = []
    if test_data is not None:
        # Preprocess the test data
        categorical_columns = test_data.select_dtypes(include=['object']).columns
        numerical_columns = test_data.select_dtypes(include=['float64']).columns

        for col in categorical_columns:
            test_data[col] = label_encoder.fit_transform(test_data[col])

        test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])

        total_rows = len(test_data)

        # Classify each row
        for i, row in test_data.iterrows():
            input_data = row.to_numpy().reshape(1, -1)

            # Make predictions using the CNN model
            cnn_probabilities = cnn_model.predict(input_data)
            cnn_predictions = np.argmax(cnn_probabilities, axis=-1)

            # Make predictions using the SVM model
            svm_predictions = svm_model.predict(input_data)

            # Convert numerical labels back to original labels
            if int(cnn_predictions[0])==0:
                cnn_label="normal"
            if int(cnn_predictions[0])==1:
                cnn_label="suspicious"
            if int(cnn_predictions[0])==2:
                cnn_label="unknown"
            #cnn_label = label_encoder.inverse_transform([cnn_predictions[0]])[0]
            svm_label = str(svm_predictions[0])

            cnn_results.append(cnn_label)
            svm_results.append(svm_label)

            # Calculate and update the loading percentage
            loading_percentage = (i + 1) * 100 / total_rows
            loading_var.set(f"Classifying... {loading_percentage:.2f}%")

    else:
        loading_var.set("No test data to classify.")

    # Update the results display
    display_results()

# Create a function to open the 'test.csv' file
def open_test_file():
    global test_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        test_data = pd.read_csv(file_path)

# Create a function to display the results
def display_results():
    results_window = tk.Toplevel(root)
    results_window.title("Classification Results")

    frame = ttk.Frame(results_window)
    frame.grid(row=0, column=0)

    tree = ttk.Treeview(frame, columns=["CNN Predictions", "SVM Predictions"], show="headings")
    tree.heading("CNN Predictions", text="CNN Predictions")
    tree.heading("SVM Predictions", text="SVM Predictions")
    tree.column("CNN Predictions", width=200)
    tree.column("SVM Predictions", width=200)

    for i in range(len(cnn_results)):
        tree.insert("", "end", values=[cnn_results[i], svm_results[i]])

    tree.grid(row=0, column=0)

# Create the main application window
root = tk.Tk()
root.title("Multi-Model Classification")

# Create a button to open the 'test.csv' file
open_file_button = ttk.Button(root, text="Open Test File", command=open_test_file)
open_file_button.pack(pady=10)

# Create a button to start classification
classify_button = ttk.Button(root, text="Classify Test Data", command=lambda: threading.Thread(target=classify_test_data).start())
classify_button.pack(pady=10)

# Create a variable to display loading percentage
loading_var = tk.StringVar()
loading_label = ttk.Label(root, textvariable=loading_var)
loading_label.pack(pady=10)

root.mainloop()
