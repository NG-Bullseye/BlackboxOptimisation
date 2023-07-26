import csv
def main(file_path):
    # Reading from csv file
    results_loaded = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            results_loaded[row[0]] = float(row[1])

    print(results_loaded)
    return results_loaded