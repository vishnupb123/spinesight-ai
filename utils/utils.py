import csv
import os

# Define the path to the history.csv file
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'user_data', 'user_data.csv')

# Function to fetch historical records from the CSV file
def get_records():
    records = []
    try:
        with open(CSV_FILE_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                records.append(row)
    except FileNotFoundError:
        print("The history file does not exist.")
    return records
