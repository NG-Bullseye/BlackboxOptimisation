import random
import numpy as np
import time
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt

from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
maxiter_list = []
average_optimal_fxs = []


def run_BO_multiple_iterations(min_iter, max_iter, plotting, step=1):
    for maxiter in range(min_iter, max_iter + 1, step):
        print(f"Running BO with maxiter = {maxiter}")
        avg_optimal_fx = BO(plotting)
        print(f"Appending {avg_optimal_fx} to average_optimal_fxs")

        average_optimal_fxs.append(avg_optimal_fx)
        maxiter_list.append(maxiter)

        print(f"Current maxiter_list: {maxiter_list}")
        print(f"Current average_optimal_fxs: {average_optimal_fxs}")

    plot_performance(max_iter)
def plot_samples_old():
    yaw_acc = app.sampler.yaw_acc
    x_values = list(yaw_acc.keys())
    x_values = [float(x) for x in x_values]

    #print("Shape of x_values:", np.shape(x_values))
    #print("Types in x_values:", [type(x) for x in x_values])

    y_values = list(yaw_acc.values())
    samples=app.sampler.sampled_values_for_vanilla_bo
    x_sampels =samples
    x_sampels = [x[0] if isinstance(x, np.ndarray) else x for x in x_sampels]

    #print("x_sampels before f_discrete_real_data call:", x_sampels)

    y_samepls = app.sampler.f_discrete_real_data(x_sampels)
    plt.scatter(app.sampler.shift_to_original(x_sampels) , y_samepls, color='red', label='Sampled Points')
    plt.scatter(x_values, y_values, color='blue', label='obj func')
    plt.title("Sampled Points Over Objective Function Values")
    plt.xlabel("x-values")
    plt.ylabel("Objective Function Values")
    plt.legend()
    plt.show()


def plot_performance(maxiter):
    global_max = app.sampler.getGlobalOptimum_Y()
    percentage_close_list = [(fx / global_max) * 100 for fx in average_optimal_fxs]

    print(len(maxiter_list), len(percentage_close_list))

    plt.plot(maxiter_list, percentage_close_list, marker='o')
    if maxiter==0:
        maxiter=1
    # Set axis limits
    plt.xlim(0, maxiter)
    plt.ylim(0, 100)  # percentage can be up to 100

    # Ensure x-axis ticks are integers
    plt.xticks(np.arange(0, maxiter + 1, step=1))

    plt.title("Performance of BO for different max iterations (Percentage Close to Global Max)")
    plt.xlabel("Max Iterations")
    plt.ylabel("Percentage Close to Global Max")
    plt.show()


def plot_samples():
    yaw_acc = app.sampler.yaw_acc
    x_values = list(yaw_acc.keys())
    x_values = [float(x) for x in x_values]

    #print("Shape of x_values:", np.shape(x_values))
    #print("Types in x_values:", [type(x) for x in x_values])

    y_values = list(yaw_acc.values())
    #print("Before sorting: ", x_values)

    # Sort the x_values and y_values based on the sorted order of x_values
    sorted_pairs = sorted(zip(x_values, y_values))
    x_values, y_values = zip(*sorted_pairs)
    #print("After sorting: ", x_values)

    samples = app.sampler.sampled_values_for_vanilla_bo
    x_samples = samples
    x_samples = [x[0] if isinstance(x, np.ndarray) else x for x in x_samples]

    #print("x_samples before f_discrete_real_data call:", x_samples)

    y_samples = app.sampler.f_discrete_real_data(x_samples)

    # Plotting sampled points in red
    plt.scatter(app.sampler.shift_to_original(x_samples), y_samples, color='red', label='Sampled Points')

    # Plotting objective function values in blue as a line
    plt.plot(x_values, y_values, color='blue', label='obj func')

    plt.title("Sampled Points Over Objective Function Values")
    plt.xlabel("x-values")
    plt.ylabel("Objective Function Values")

    # Setting y-axis limits
    plt.ylim(0, 1)

    plt.legend()
    plt.show()

def BO(plotting):
    # Bayesian Optimization
    for i in range(n_repeats):
        app.sampler.sampled_values_for_vanilla_bo=[]
        np.random.seed(random.seed(int(time.perf_counter() * 1e9)))
        all_evals.clear()
        app.sampler.regrets=[] #reset regrets
        optimizer = BayesianOptimization(
            f=app.sampler.f_discrete_real_data_x,
            pbounds={'x': (0, 90)},
            random_state=1,
        )
        initial_point_x = np.random.uniform(0,90, 1)[0] # replace with the x0 you want
        print(f"initial_point_x:{app.sampler.shift_to_original( initial_point_x)}")

        initial_point_y = app.sampler.f_discrete_real_data_x(initial_point_x)  # replace with your function
        print(f"initial_point_y:{initial_point_y}")

        optimizer.register(params={'x': initial_point_x}, target=initial_point_y)
        start_time = time.time()

        optimizer.maximize(
            init_points=0,
            n_iter=maxiter
        )
        result = optimizer.max
        end_time = time.time()
        time_taken = end_time - start_time
        n_eval = app.sampler.function_call_counter
        app.sampler.function_call_counter = 0
        optimal_x, optimal_fx = result['params']['x'], result['target']
        cum_regret = np.sum(app.sampler.regrets)/len(app.sampler.regrets)
        cum_regrets.append(cum_regret)
        optimal_xs.append(optimal_x)
        optimal_fxs.append(optimal_fx)
        n_evals.append(n_eval)
        times.append(time_taken)
        if plotting:
            plot_samples()
    print("cum_regrets:", cum_regrets)
    print("optimal_xs:", optimal_xs)
    print("optimal_fxs:", optimal_fxs)
    print("times:", times)
    print("n_evals:", n_evals)
    # Debugging: Check the initial state
    #print(f"Obj func: {app.sampler.yaw_acc}")
    for arr in [cum_regrets, optimal_xs, optimal_fxs, times, n_evals]:
        print(np.array(arr).shape)
    avg_values = list(
        map(lambda x: np.mean(list(map(np.mean, x))), [cum_regrets, optimal_xs, optimal_fxs, times, n_evals]))
    global_optima_key, global_optima_value = app.sampler.getGlobalOptimum_X(),app.sampler.getGlobalOptimum_Y() #max(obj_func.items(), key=lambda x: x[1])
    print(f"Average Cumulative Regret over {maxiter} runs: {avg_values[0]}")
    print(f'Global optima: x={global_optima_key} y={global_optima_value}')
    print(f"Found optima: x={app.sampler.shift_to_original(avg_values[1])} y={avg_values[2]}")
    print(f"With {avg_values[4]} Evaluations")
    print(f"Average time taken: {avg_values[3]}")
    avg_optimal_fx = np.mean(optimal_fxs[-n_repeats:])
    return avg_optimal_fx

if __name__ == '__main__':
    np.random.seed(random.seed(int(time.perf_counter() * 1e9)))
    app = Application(Sampler(0.1, Sim()))
    app.sampler.reset()
    bounds = [(0, 90)]
    n_repeats = 10
    maxiter = 0
    obj_func = app.sampler.shift_dict_to_positive(app.sampler.yaw_acc)
    all_evals = []
    cum_regrets, optimal_xs, optimal_fxs, n_evals, times = [], [], [], [], []
    run_BO_multiple_iterations(min_iter=0, max_iter=maxiter, step=1,plotting=False)

