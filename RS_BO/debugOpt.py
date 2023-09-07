
import random
import numpy as np
from scipy.optimize import minimize
import time
import cma
from bayes_opt import BayesianOptimization
from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim
if __name__ == '__main__':
    np.random.seed(random.seed(int(time.perf_counter() * 1e9)))
    app = Application(Sampler(0.1, Sim()))
    app.sampler.reset()
    bounds = [(0, 90)]
    n_repeats = 20
    maxiter = 10
    obj_func = app.sampler.shift_dict_to_positive(app.sampler.yaw_acc)
    all_evals = []
    cum_regrets, optimal_xs, optimal_fxs, n_evals, times = [], [], [], [], []

    def callback(x):
        all_evals.append(app.sampler.f_discrete_counter(x))


#    # Powell
#    for i in range(n_repeats):
#        all_evals.clear()
#        start_time = time.time()
#        x0 = np.random.uniform(0, 90)
#        result = minimize(app.sampler.f_discrete_counter, x0, bounds=bounds, method='Powell',
#                          options={'maxiter': maxiter},
#                          callback=callback)
#        end_time = time.time()
#        time_taken = end_time - start_time
#        n_eval = app.sampler.function_call_counter
#        app.sampler.function_call_counter = 0
#        optimal_x, optimal_fx = result.x, -result.fun
#        cum_regrets.append(np.sum(np.max(list(obj_func.values())) - np.array(all_evals)))
#        optimal_xs.append(optimal_x)
#        optimal_fxs.append(optimal_fx)
#        n_evals.append(n_eval)
#        times.append(time_taken)
#
#    # Nelder-Mead
#    for i in range(n_repeats):
#        all_evals.clear()
#        start_time = time.time()
#        x0 = np.random.uniform(0, 90)
#        result = minimize(app.sampler.f_discrete_counter, x0, bounds=bounds, method='Nelder-Mead',
#                          options={'maxiter': maxiter},
#                          callback=callback)
#        end_time = time.time()
#        time_taken = end_time - start_time
#        n_eval = app.sampler.function_call_counter
#        app.sampler.function_call_counter = 0
#        optimal_x, optimal_fx = result.x, -result.fun
#        cum_regrets.append(np.sum(np.max(list(obj_func.values())) - np.array(all_evals)))
#        optimal_xs.append(optimal_x)
#        optimal_fxs.append(optimal_fx)
#        n_evals.append(n_eval)
#        times.append(time_taken)
#
    # Bayesian Optimization
    for i in range(n_repeats):
        all_evals.clear()
        start_time = time.time()
        optimizer = BayesianOptimization(
            f=app.sampler.f_discrete_counter,
            pbounds={'x': (0, 90)},
            random_state=1,
        )
        optimizer.maximize(
            init_points=2,
            n_iter=maxiter
        )
        result = optimizer.max
        end_time = time.time()
        time_taken = end_time - start_time
        n_eval = app.sampler.function_call_counter
        app.sampler.function_call_counter = 0
        optimal_x, optimal_fx = result['params']['x'], -result['target']
        cum_regrets.append(np.sum(np.max(list(obj_func.values())) - np.array(all_evals)))
        optimal_xs.append(optimal_x)
        optimal_fxs.append(optimal_fx)
        n_evals.append(n_eval)
        times.append(time_taken)

    print("cum_regrets:", cum_regrets)
    print("optimal_xs:", optimal_xs)
    print("optimal_fxs:", optimal_fxs)
    print("times:", times)
    print("n_evals:", n_evals)

    for arr in [cum_regrets, optimal_xs, optimal_fxs, times, n_evals]:
        print(np.array(arr).shape)
    avg_values = list(
        map(lambda x: np.mean(list(map(np.mean, x))), [cum_regrets, optimal_xs, optimal_fxs, times, n_evals]))
    global_optima_key, global_optima_value = max(obj_func.items(), key=lambda x: x[1])
    print(f"Average Cumulative Regret over {maxiter} runs: {avg_values[0]}")
    print(f'Global optima: x={global_optima_key} y={global_optima_value}')
    print(f"Found optima: x={avg_values[1]} y={avg_values[2]}")
    print(f"With {avg_values[4]} Evaluations")
    print(f"Average time taken: {avg_values[3]}")



