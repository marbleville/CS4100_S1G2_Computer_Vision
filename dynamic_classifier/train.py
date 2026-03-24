import csv
import os

def extract_data(file_path):
    out = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            out.append([int(num) for num in row])
    return out

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
right_data = extract_data(os.path.join(BASE_DIR, "dynamic_classifier", "right.csv"))
print(right_data)