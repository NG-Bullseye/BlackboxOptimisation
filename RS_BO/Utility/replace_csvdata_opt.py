import csv

final_gridsearch_results =[0.861066666669,0.8808199999999982, 0.8917899999999943, 0.8982599999999923, 0.9021399999999921, 0.9051499999999901, 0.9072099999999924, 0.9086799999999925, 0.9095799999999933, 0.9123499999999937, 0.9136599999999938, 0.9133799999999942, 0.9142999999999953, 0.9169999999999955, 0.9161399999999958, 0.9180599999999975, 0.9180599999999964, 0.9203899999999989, 0.9199099999999993, 0.9218999999999992, 0.9215499999999998, 0.9236800000000005, 0.9228599999999992, 0.9245400000000007, 0.925860000000003, 0.926290000000003, 0.9258000000000031, 0.9277500000000032, 0.9296400000000046, 0.9309300000000046, 0.9317200000000058]


csv_file_path = "/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/OptimalFX_PlottingData_n_repeats_300_iterations_31.csv"  # Replace this with the path to your CSV file

# Read existing rows
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    rows = [row for row in reader]

# Update rows with new grid search results
for i, result in enumerate(final_gridsearch_results):
    rows[i]['Sim_An_OptimalFX'] = result

# Write updated rows back to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("CSV file updated.")