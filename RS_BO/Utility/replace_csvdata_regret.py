import csv

final_gridsearch_results =[0.091,0.17774, 0.2670700000000006, 0.3568699999999994, 0.44735999999999987, 0.5383299999999998, 0.6288099999999995, 0.7176599999999981, 0.8150500000000007, 0.8953599999999995, 0.9917299999999994, 1.0930100000000014, 1.185439999999999, 1.2581099999999996, 1.3693000000000017, 1.4553499999999973, 1.5520699999999998, 1.6295400000000013, 1.7308300000000003, 1.8227500000000016, 1.9132000000000013, 1.9972800000000022, 2.1020399999999992, 2.1880699999999984, 2.276910000000003, 2.367589999999996, 2.4720299999999966, 2.556329999999998, 2.644040000000001, 2.7279800000000027, 2.8192200000000045]




csv_file_path = \
    "/home/lwecke/Github/BlackboxOptimisation/RS_BO/Benchmark/Regrets_PlottingData_n_repeats_300_iterations_31.csv"  # Replace this with the path to your CSV file

# Read existing rows
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    rows = [row for row in reader]

# Update rows with new grid search results
for i, result in enumerate(final_gridsearch_results):
    rows[i]['Sim_An_Regrets'] = result

# Write updated rows back to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("CSV file updated.")