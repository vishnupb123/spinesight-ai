import os
import csv
from datetime import datetime

CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'user_data')
os.makedirs(CSV_FOLDER, exist_ok=True)  

CSV_PATH = os.path.join(CSV_FOLDER, 'user_data.csv')
def log_prediction_to_csv(input_data, prediction, confidence, message, csv_path=CSV_PATH):
    print(f"Logging prediction to {csv_path}...")
    fieldnames = list(input_data.keys()) + ['prediction', 'confidence', 'ai_message', 'timestamp']

    # Add timestamp
    input_data_with_output = input_data.copy()
    input_data_with_output.update({
        'prediction': prediction,
        'confidence': confidence,
        'ai_message': message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only once
        if not file_exists:
            writer.writeheader()

        writer.writerow(input_data_with_output)
