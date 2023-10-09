import csv
from scipy.optimize import minimize
import numpy as np
from RS_BO.Custom_Gaussian import Optimization as opt
from datetime import datetime
from RS_BO.Utility.Sim import Sim
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Qt5Agg') # Make sure you have PyQt5 or PySide2 installed
import matplotlib.pyplot as plt
import os
class PolynomialPredictor:
    def __init__(self, csv_file):
        with open(csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            coeff = [float(x) for x in next(csv_reader)]
            self.intercept = coeff[0]
            self.coefficients = np.array(coeff[1:])
    def fn(self, x):
        # Initialize variables to keep track of polynomial elements and index
        x_poly_elements = []
        coeff_length = len(self.coefficients)
        # Loop over each coefficient index
        for i in range(coeff_length):
            # Calculate the value of x raised to the power of i
            x_to_the_i = x ** i
            # Add this value to the list of polynomial elements
            x_poly_elements.append(x_to_the_i)
        # Convert the list of polynomial elements to a NumPy array
        x_poly = np.array(x_poly_elements)
        # Calculate the dot product of the coefficients and the polynomial elements
        weighted_sum = np.dot(self.coefficients, x_poly)
        # Retrieve the intercept from the object's attributes
        intercept_value = self.intercept
        # Calculate the final value of the polynomial equation
        y_poly = intercept_value + weighted_sum
        if (y_poly>1).any() and len(self.coefficients)> 6:
            print(f"not good y_poly > 1 {y_poly} and len(self.coefficients) is {len(self.coefficients)}")
        return y_poly
class PolynomialPredictor_Acc(PolynomialPredictor):
    def __init__(self, csv_file):
        super().__init__(csv_file)
class PolynomialPredictor_Rec(PolynomialPredictor):
    def __init__(self, csv_file):
        super().__init__(csv_file)

class Sampler():
    def __init__(self, dataObj):
        self.evaluated_x = []
        self.evaluated_fx = []
        self.sampled_values_for_vanilla_bo = []
        self.regrets = np.array([])
        self.function_call_counter = 0
        self.QUANTIZATION_FACTOR = 0.1
        self.shift_value = -45  # set to the minimum of the negative value interval of sampled data
        self.dataObj = dataObj
        self.yaw_acc = self.dataObj.yaw_acc_mapping
        self.yaw_rec = self.dataObj.yaw_vec_mapping
        self.yaw_list = self.dataObj.yaw_list
        self.fx_max = None
        self.fx_min = None
        self.fx_max_computed = False
        self.fx_min_computed = False
        self.acc_predictor=PolynomialPredictor_Acc(f'/home/lwecke/PycharmProjects/flow_regime_recognition_CameraPosition/modules/Bulktrain/fittedPolynomial_acc.csv')
        self.rec_predictor=PolynomialPredictor_Rec(f'/home/lwecke/PycharmProjects/flow_regime_recognition_CameraPosition/modules/Bulktrain/fittedPolynomial_rec.csv')
        self.find_extrema()  # Find extrema during object initialization
        self.max_rec=150
        self.min_rec=-150
        self.findMaxima= True

    def objective_function(self, x, sign=1.0):
        x = np.atleast_1d(x)
        return sign * self.sample_continues_not_normalized(x)

    def find_extrema(self):

        if self.findMaxima:
            x_vals = np.linspace(-45, 45, 100000)  # 10,000 grid points
            f_vals_max = np.array([-self.objective_function([x], -1.0) for x in x_vals])
            f_vals_min = np.array([self.objective_function([x], 1.0) for x in x_vals])

            print(f"Debug Max Value in f_vals_max: {np.max(f_vals_max)}")
            print(f"Debug Min Value in f_vals_min: {np.min(f_vals_min)}")

            try:
                index_max = np.argmax(f_vals_max)
                self.fx_max = f_vals_max[index_max][0]
                print(f"Maxima found at x = {x_vals[index_max]} with value = {self.fx_max}")

                index_min = np.argmin(f_vals_min)
                self.fx_min = f_vals_min[index_min][0]
                print(f"Minima found at x = {x_vals[index_min]} with value = {self.fx_min}")

            except Exception as e:
                print(f"Exception occurred during optimization: {e}")

            print(f"Computed fx_max: {self.fx_max}")
            print(f"Computed fx_min: {self.fx_min}")

            if self.fx_max == self.fx_min:
                x_vals = np.linspace(-45, 45, 1000)
                y_vals = [self.objective_function([x], 1.0) for x in x_vals]
                plt.plot(x_vals, y_vals)
                plt.title("Test plot")
                plt.show()
                raise Exception("Warning: fx_max and fx_min are equal. This is not expected.")
        else: #set maximum manualy to speed up runtime
            self.fx_max =0.8579445786858448
            self.fx_min = 0.8193933891878757

    def sample_continues(self, x):
        if self.fx_max is None or self.fx_min is None:
            raise Exception("fx_max and fx_min cannot be None, recheck function extrema.")

        y_value = self.sample_continues_not_normalized(x)
        if y_value > self.fx_max or y_value < self.fx_min:pass
            #raise Exception(f"{y_value} > {self.fx_max} or {y_value} < {self.fx_min} = False")
        normalized_y = (y_value - self.fx_min) / (self.fx_max - self.fx_min)
        if normalized_y > 1 or normalized_y < 0:pass
            #raise Exception(f"{normalized_y} > 1 or {normalized_y} < 0 = False")
        return normalized_y
    def sample_continues_not_normalized(self, x_values):
        y_values = self.acc_predictor.fn(x_values)
        return y_values

    import numpy as np
    def compute_metric(self, alpha=0.8, beta=2.0):
        x_train=self.evaluated_x
        n = len(x_train)
        if n < 2:
            return 1  # Not enough elements to compute metric
        metric = 0.0
        total_weight = 0.0

        for i in range(1, n):
            weight = (alpha ** (n - i)) * (beta ** (i - n + 1))
            diff = abs(x_train[i] - x_train[i - 1])
            metric += weight * diff
            total_weight += weight
        # Normalize the metric
        metric /= total_weight
        # Compute the closeness metric, where 1 means very close and 0 means not close
        closeness_metric =  1 / (1 + metric)

        return closeness_metric
    def get_recscalar_of_x_continues(self, x_values):
        self.evaluated_fx.append(x_values[0])
        rec_scalar = self.rec_predictor.fn(x_values)

        def nonlinear_scale(value):
            scaled_value = np.sign(value) * np.power(np.abs(value), 0.5)
            scaled_value[np.abs(scaled_value) < 50] = np.sign(scaled_value[np.abs(scaled_value) < 50]) * np.power(
                np.abs(scaled_value[np.abs(scaled_value) < 50]), 3)
            return scaled_value

        rec_scalar_scaled = nonlinear_scale(rec_scalar)*self.compute_metric( alpha=0.8, beta=2.0)
        return rec_scalar_scaled
    def sample(self, yaw_string):
        raise Exception("DEPRECIATED")
    def get_closest_key(self,d, value, tolerance=1e-2):
        closest_key = None
        closest_distance = float('inf')

        for key in d.keys():
            distance = key - value
            if abs(distance) < tolerance and abs(distance) < abs(closest_distance):
                closest_key = key
                closest_distance = distance
        return closest_key
    def return_zero(self,x):
        return 0
    def x_discrete(self,x):
        return np.round(x / self.QUANTIZATION_FACTOR) * self.QUANTIZATION_FACTOR
    def f_discrete(self,x):
        sampled_acc = (np.sin(self.x_discrete(x)) + 1) / 2
        return sampled_acc
    def f_discrete_counter(self,x):
        self.function_call_counter += 1
        return self.f_discrete(x)
    def reset(self):
        self.function_call_counter = 0

    def f_discrete_real_data(self, shifted_yaws):
        self.function_call_counter += 1
        sampled_accs = []
        original_yaws=self.shift_to_original(shifted_yaws)
        for yaw in original_yaws:
            yaw_value = self.x_discrete_real_data_unshifted([yaw])# Get the single value from the returned list
            yaw_value = yaw_value[0]
            sampled_acc = self.sample_continues(yaw_value)
            sampled_accs.append(sampled_acc)
        return sampled_accs

    def calculateRegret(self, y_array):
        y_array = np.array(y_array)
        regret = np.abs(self.getGlobalOptimum_Y() - y_array)
        self.regrets=np.append(self.regrets,regret)
        return regret
    def getPreviousRegret(self):
        return self.getPreviousRegret()
    def getGlobalOptimum_X(self):
        Exception("not Implemented")
    def getGlobalOptimum_Y(self):
        return self.fx_max
    def getGlobalMin_Y(self):
        return self.fx_min
    def f_discrete_real_data_x(self, x):
        self.function_call_counter += 1
        sampled_accs = []

        original_yaws = self.shift_to_original(x)
        original_yaws = np.atleast_1d(original_yaws)

        #print("Original yaws: ", original_yaws)
        if original_yaws.size == 0:
            print("original_yaws is empty.")
            return 0.0  # return a default value

        for yaw in original_yaws:
            yaw_value = self.x_discrete_real_data_unshifted([yaw])  # Get the single value from the returned list
            yaw_value = yaw_value[0]
            sampled_acc = self.sample_continues(yaw_value)

            sampled_accs.append(sampled_acc)

        self.calculateRegret(sampled_accs)#hier muss regret berechnet werden! damit der in debugopt2 angegeben werden kann über das app.sampler obj
        self.sampled_values_for_vanilla_bo.append(self.x_discrete_real_data(yaw_value))
        self.evaluated_fx.append(sampled_accs[0])
        return sampled_accs[0]

    def x_discrete_real_data_unshifted(self, yaws):
        if not isinstance(yaws, (np.ndarray, list, tuple)):
            yaws = np.array([yaws])
        return yaws

    def x_discrete_real_data(self, yaws):
        if not isinstance(yaws, (np.ndarray, list, tuple)):
            yaws = np.array([yaws])
        return yaws

    def offset_scalar_real_data(self, x):
        x_shifted = self.shift_to_original(x)
        rec_scalar_list = self.get_recscalar_of_x_continues(x_shifted)
        rec_scalar = np.array(rec_scalar_list)  # Converting list to numpy array
        return rec_scalar

    def shift_dict_to_positive(self,d):
        min_key = min(d.keys())
        if min_key < 0:
            shift_value = abs(min_key)
            d = {key + shift_value: value for key, value in d.items()}
        return d

    def shift_to_positive(self, x_values):
        x_values = np.array(x_values)  # Convert input to numpy array
        if self.shift_value < 0:  # Just check the scalar value directly
            x_values -= self.shift_value  # Shift all values to positive
        return x_values

    def shift_to_original_BO(self, x_values):
        if self.shift_value is None:
            raise ValueError("shift_value has not been set. Please run shift_to_positive first.")
        x_values = np.atleast_1d(x_values)  # Ensure it is at least 1-D
        if self.shift_value < 0:
            x_values += self.shift_value  # Reverse the shift to get original values
        return x_values

    def shift_to_original(self, x_values):
        if self.shift_value is None:
            raise ValueError("shift_value has not been set. Please run shift_to_positive first.")

        x_values = np.array(x_values)  # Convert to numpy array for consistency
        if self.shift_value < 0:
            x_values += self.shift_value  # Reverse the shift to get original values
        return x_values

class Application():
    def __init__(self, sampler):
        self.sampler=sampler
        self.plotting = False
        self.shift_value = 0
        self.randomseed=455
        self.QUANTIZATION_FACTOR = 1
        self.INTERVAL = 45
        self.ITERATIONS = 5

        #results
        self.optimal_x = None #results of the Optimisation
        self.optimal_fx = None #results of the Optimisation
        self.x_train_sorted_og =None #results of the Optimisation
        self.y_train_sorted = None  #results of the Optimisation
    def reset_f_discrete_counter_counter(self):
        self.f_discrete_counter_counter = 0
    def kernel(self,a, b, l=1.0):
        sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sqdist / l**2)
    def plot_gp(self,gp, x_test, f_discrete):
        mu_star, var_star = gp.mu_star, gp.var_star
        x_train, y_train = gp.x_train, gp.y_train

        plt.figure(figsize=(12, 8))
        plt.plot(x_test, f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
        plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
        plt.plot(x_test, mu_star, 'b-', label='Prediction')
        plt.fill(np.concatenate([x_test, x_test[::-1]]),
                 np.concatenate([mu_star - 1.9600 * var_star,
                                 (mu_star + 1.9600 * var_star)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend(loc='upper left')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.title("Offset - Created at: " + current_time)
        plt.show()

    def start_sim_with_test_data(self):
        optimizer = opt(n_iterations=self.ITERATIONS, quantization_factor=1, offset_range=5, offset_scale=0.1,
                     kernel_scale=3, protection_width=1)

        x_train = self.sampler(np.random.uniform(0, self.INTERVAL, 1))

        y_train = self.sampler.f_discrete(x_train)

        x_test = np.linspace(0, self.INTERVAL, 100)

        x_train, y_train, mu_star, var_star = optimizer.optimize(x_train, y_train, x_test, self.sampler.f_discrete, self.sampler.x_discrete)

        #self.benchmark(self.INTERVAL,self.ITERATIONS)

        plt.figure(figsize=(12, 8))
        plt.plot(x_test, self.sampler.f_discrete(x_test), 'r:', label=r'$f(x) = \frac{\sin(x) + 1}{2}$')
        plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
        plt.plot(x_test, mu_star, 'b-', label='Prediction')
        plt.fill_between(x_test, mu_star - 1.9600 * var_star, mu_star + 1.9600 * var_star, color='b', alpha=.5,
                         label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend(loc='upper left')
        plt.title("Offset")
        plt.show()
    def callback_plotting(self,x_test,x_train_pos,y_train_pos,mu_star,var_star):
        # Your existing plotting code with modifications
        # Your existing sorting code
        x_test_sorted = x_test
        x_train_sorted = x_train_pos
        self.y_train_sorted = y_train_pos
        mu_star_sorted = mu_star
        var_star_sorted = var_star
        # Create interpolation functions
        x_new = np.linspace(min(x_test_sorted), max(x_test_sorted), 300)  # 300 new x-points for interpolation

        # Interpolation for upper and lower bounds
        upper_bound = mu_star_sorted + 1.9600 * var_star_sorted
        lower_bound = mu_star_sorted - 1.9600 * var_star_sorted
        f_upper = interp1d(x_test_sorted, upper_bound, kind='cubic')
        f_lower = interp1d(x_test_sorted, lower_bound, kind='cubic')

        # Interpolation for prediction
        f_mu_star = interp1d(x_test_sorted, mu_star_sorted, kind='cubic')

        # Interpolation for Sampled Data
        y_test = np.array(self.sampler.f_discrete_real_data(x_test_sorted))
        f_y_test = interp1d(x_test_sorted, y_test, kind='cubic')

        # Interpolated upper, lower bounds, prediction, and Sampled Data
        upper_bound_smooth = f_upper(x_new)
        lower_bound_smooth = f_lower(x_new)
        mu_star_smooth = f_mu_star(x_new)
        y_test_smooth = f_y_test(x_new)
        x_test_sorted_og = self.sampler.shift_to_original(x_test_sorted)
        self.x_train_sorted_og = self.sampler.shift_to_original(x_train_sorted)
        x_new_og = self.sampler.shift_to_original(x_new)
        # Your existing plotting code with modifications
        plt.figure(figsize=(12, 8))
        x_ticks = np.linspace(-45, 45, 20)
        plt.xticks(x_ticks)

        plt.ylim(-2, 2)

        # Plotting the smoothed Sampled Data curve
        plt.plot(x_new_og, y_test_smooth, 'r:', label='Sampled Data')

        plt.plot(self.x_train_sorted_og, self.y_train_sorted, 'r.', markersize=10, label='Observations')

        # Plotting the smoothed prediction curve
        plt.plot(x_new_og, mu_star_smooth, 'b-', label='Prediction')

        # Using the smoothed 95% confidence interval
        plt.fill_between(x_new_og, lower_bound_smooth, upper_bound_smooth, color='b', alpha=0.5,
                         label='95% confidence interval')

        plt.xlabel('$yaw$')
        plt.ylabel('$acc$')
        plt.legend(loc='upper left')
        plt.show()

    def start_sim_with_real_data(self, quantization_factor=1., offset_range=1., offset_scale=1.,
                        kernel_scale=1.3, protection_width=1.,n_iterations=10,randomseed=42,deactivate_rec_scalar=False,plotting=False):
        self.randomseed=randomseed
        optimizer = opt(self.sampler.offset_scalar_real_data,quantization_factor, offset_range, offset_scale,
                            kernel_scale, protection_width, n_iterations,self.callback_plotting,plotting=plotting,deactivate_rec_scalar=deactivate_rec_scalar)

        x_train_pos = self.sampler.shift_to_positive(self.initialPoint())
        y_train_pos = np.array(self.sampler.f_discrete_real_data(x_train_pos))
        x_test = np.linspace(-45, 45, 1000)
        x_test = self.sampler.shift_to_positive(x_test)

        x_train_pos, y_train_pos, mu_star, var_star = optimizer.optimize(
            x_train_pos, y_train_pos, x_test, self.sampler.f_discrete_real_data, self.sampler.x_discrete_real_data
        )

        # Find the optimal solution
        optimal_index = np.argmax(y_train_pos)
        optimal_x = x_train_pos[optimal_index]
        optimal_fx = y_train_pos[optimal_index]

        cumulative_regret = optimizer.get_cumulative_regret()

        self.optimal_x=optimal_x
        self.optimal_fx=optimal_fx
        return cumulative_regret

    def initialPoint(self):
        np.random.seed(self.randomseed)
        x_train = self.sampler.x_discrete_real_data_unshifted(np.random.uniform(-45, self.INTERVAL, 1))
        return x_train
    def test_initial_point_destib(self):
        avg_x_train=0.
        for i in range(1000):
            avg_x_train +=  self.initialPoint()[0]
        avg_x_train /= 1000
        print(self.sampler.shift_to_positive(avg_x_train))

    def plot_sampled_continuous_function(self):
        # Generate 1000 points between -45 and 45
        x_values = np.linspace(-45, 45, 1000)

        # Get the y-values from the continuous function
        y_values = self.sampler.get_recscalar_of_x_continues(x_values)

        # Create the plot
        plt.figure()
        plt.plot(x_values, y_values, label='Sampled Continuous Function')
        plt.xlabel('X-values')
        plt.ylabel('Y-values')
        plt.title('Plot of Sampled Continuous Function')
        plt.legend()
        plt.grid(True)
        plt.show()

    def objective_function(self, x):
        # Make sure x is a 1D array
        x = np.atleast_1d(x)
        #xp = self.sampler.shift_to_original(x)
        return -self.sampler.sample_continues(x)

    def find_optimum(self):
        # Initial guess (starting point for optimization)
        x0 = np.array([0.0])

        # Bounds for x
        bounds = [(-45, 45)]

        # Perform the optimization
        result = minimize(self.objective_function, x0, bounds=bounds, method='L-BFGS-B')

        if result.success:
            optimized_x = result.x
            optimized_y = -result.fun  # Convert back to positive as we had negated the value
            return optimized_x, optimized_y
        else:
            raise Exception("Optimization failed: " + result.message)

def main(dbName):
    app = Application(Sampler(Sim(dbName)))
    b=os.environ.get("test")
    print("Environment Variable:", b)  # Debug print
    if b=='1':
        app.start_sim_with_test_data()
    if b=='0':
        app.start_sim_with_real_data(deactivate_rec_scalar=True,plotting=True,quantization_factor=1.1,
                                     kernel_scale=0.27, offset_scale= 3.0,offset_range=10., protection_width=10.
                                     ,n_iterations=10,randomseed= 524)


if __name__ == '__main__':
    dbName="Testdata"
    app = Application(Sampler(Sim(dbName)))
    params = {
        'quantization_factor': 1,
        'kernel_scale':    6.021461291655982  ,
        'offset_scale':    0.003+0.003,
        'offset_range':    0.05+0.1  ,
        'protection_width':5 ,
        'n_iterations': 20,
        'randomseed': 144,
        'deactivate_rec_scalar': False,
        'plotting': True
    }
    app.start_sim_with_real_data(**params)
