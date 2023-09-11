import random
import numpy as np
import time
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt

from Application import Application, Sampler
from RS_BO.Utility.Sim import Sim


class BaysianOptimization:
    def __init__(self,app,maxiter,n_repeats):
        self.maxiter_list = []
        self.average_cum_reg = []

        self.app = app
        self.cum_regrets = []
        self.optimal_xs = []
        self.optimal_fxs = []
        self.n_evals = []
        self.times = []
        self.maxiter=maxiter
        self.n_repeats=n_repeats
        self.average_optimal_fxs = []
        self.maxiter_list = []
        np.random.seed(random.seed(int(time.perf_counter() * 1e9)))
        app.sampler.reset()
        self.bounds = [(0, 90)]
        self.obj_func = app.sampler.shift_dict_to_positive(app.sampler.yaw_acc)
        self.all_evals = []

    def run_BO_multiple_iterations(self,min_iter, max_iter, plotting, step=1):
        for maxiter in range(min_iter, max_iter + 1, step):
            print(f"Running BO with maxiter = {maxiter}")
            avg_optimal_fx = self.bo(maxiter,plotting)

            self.average_optimal_fxs.append(avg_optimal_fx)
            self.maxiter_list.append(maxiter)
            print(f"INFO: Appending {avg_optimal_fx} to average_optimal_fxs")
            print(f"CURRENT: maxiter_list: {self.maxiter_list}")
            print(f"CURRENT: average_optimal_fxs (index is iteration): {self.average_optimal_fxs}")
        if plotting:
            self.plot_performance(max_iter)
        return self.average_optimal_fxs, self.average_cum_reg

    def plot_samples_old(self):
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


    def plot_performance(self,maxiter):
        global_max = self.app.sampler.getGlobalOptimum_Y()
        percentage_close_list = [(fx / global_max) * 100 for fx in self.average_optimal_fxs]

        print(len(self.maxiter_list), len(percentage_close_list))

        plt.plot(self.maxiter_list, percentage_close_list, marker='o')
        if maxiter==0:
            maxiter=1
        # Set axis limits
        plt.xlim(0, maxiter)
        plt.ylim(0, 100)  # percentage can be up to 100

        # Ensure x-axis ticks are integers
        plt.xticks(np.arange(0, maxiter + 1, step=1))

        plt.title(f"BO")
        plt.xlabel("Max Iterations")
        plt.ylabel("Percentage Close to Global Max")
        plt.show()


    def plot_samples(self):
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

    def bo(self, maxiter, plotting):
        global cum_regrets, optimal_xs, optimal_fxs, n_evals, times
        # Reset the summary arrays
        cum_regrets, optimal_xs, optimal_fxs, n_evals, times = [], [], [], [], []
        optimal_for_this_maxiter_xs=[]
        optimal_for_this_maxiter_fxs=[]
        cum_regrets_for_this_maxiter=[]
        n_eval=0
        # Bayesian Optimization
        for i in range(self.n_repeats):
            self.app.sampler.sampled_values_for_vanilla_bo = []
            np.random.seed(random.seed(int(time.perf_counter() * 1e9)))
            self.all_evals.clear()
            self.app.sampler.regrets = []  # reset regrets
            optimizer = BayesianOptimization(
                f=self.app.sampler.f_discrete_real_data_x,
                pbounds={'x': (0, 90)},
                random_state=1,
            )

            initial_point_x = np.random.uniform(0, 90, 1)[0]
            initial_point_y = self.app.sampler.f_discrete_real_data_x(initial_point_x)
            print(f'initial_point_x:{initial_point_x} initial_point_y:{initial_point_y}')
            optimizer.register(params={'x': initial_point_x}, target=initial_point_y)
            start_time = time.time()

            optimizer.maximize(
                init_points=0,
                n_iter=maxiter
            )
            #optimal_xs.append(initial_point_x)
            #optimal_fxs.append(initial_point_y)
            result = optimizer.max
            end_time = time.time()
            time_taken = end_time - start_time
            n_eval = self.app.sampler.function_call_counter
            self.app.sampler.function_call_counter = 0

            optimal_for_this_maxiter_x, optimal_for_this_maxiter_y = result['params']['x'], result['target']
            if len(self.app.sampler.regrets)!= maxiter+1:
                print(f"ERROR  len(self.app.sampler.regrets)!= maxiter maxiter{maxiter} len(self.app.sampler.regrets){len(self.app.sampler.regrets)}")
            a=len(self.app.sampler.regrets)
            cum_regret = np.sum(self.app.sampler.regrets)

            cum_regrets_for_this_maxiter.append(cum_regret)
            optimal_for_this_maxiter_xs.append(optimal_for_this_maxiter_x)
            optimal_for_this_maxiter_fxs.append(optimal_for_this_maxiter_y)

            times.append(time_taken)
            if plotting:
                self.plot_samples()
        n_evals.append(n_eval)
        avg_cum_regrets_for_this_maxiter = np.mean(cum_regrets_for_this_maxiter[-self.n_repeats:])
        avg_optimal_for_this_maxiter_fxs = np.mean(optimal_for_this_maxiter_fxs[-self.n_repeats:])

        self.average_cum_reg.append(avg_cum_regrets_for_this_maxiter)

        global_optima_key, global_optima_value = self.app.sampler.getGlobalOptimum_X(),self.app.sampler.getGlobalOptimum_Y() #max(obj_func.items(), key=lambda x: x[1])

        print("Sampling count:", n_eval)
        print(f"Average time taken: {np.mean(times[-self.n_repeats:])}")

        print("optimal_for_this_maxiter_xs:", optimal_for_this_maxiter_xs)
        print("optimal_for_this_maxiter_fxs:", optimal_for_this_maxiter_fxs)

        print("avg_cum_regrets_for_this_maxiter:", avg_cum_regrets_for_this_maxiter)
        print("avg_optimal_for_this_maxiter_fxs:", avg_optimal_for_this_maxiter_fxs)
        print("------------------------------------------------------------------")
        print(f"AVG Found optima With {self.n_repeats} repeats and {n_eval} Iterations: y={avg_optimal_for_this_maxiter_fxs}")
        print("------------------------------------------------------------------")
        print(f'Global optima: x={global_optima_key} y={global_optima_value}')

        return avg_optimal_for_this_maxiter_fxs



def main(app,maxiter,n_repeats):
    bo=BaysianOptimization(app, maxiter, n_repeats)
    return bo.run_BO_multiple_iterations(min_iter=0, max_iter=maxiter, step=1,plotting=False)

if __name__ == '__main__':
    app = Application(Sampler(Sim()))
    print(f"FINAL RESULTS VANILLA BO: {main(app,0,1001)}")


#Average Cumulative Regret over 20 runs: 0.20757142857142857
#Global optima: x=-32.142857 y=0.76
#Found optima: x=-0.9118639582357204 y=0.627525
#With 21.0 Evaluations
#Average time taken: 2.515369304418564
#Appending 0.6309999999999999 to average_optimal_fxs
#Current maxiter_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#Current average_optimal_fxs: [0.5385, 0.5865, 0.58235, 0.59325, 0.60345, 0.59865, 0.5997, 0.627, 0.61285, 0.62995, 0.6126, 0.61765, 0.6193, 0.62655, 0.63385, 0.633, 0.6283, 0.63325, 0.63355, 0.6309999999999999]
#20 20
#
#n_evals: [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
#(200,)
#Process finished with exit code 0
#cum_regrets: [0.18857142857142856, 0.20190476190476192, 0.19714285714285718, 0.1533333333333333, 0.17476190476190476, 0.17571428571428568, 0.24333333333333332, 0.19285714285714284, 0.24761904761904763, 0.2252380952380952, 0.22333333333333336, 0.17476190476190476, 0.21904761904761907, 0.21619047619047624, 0.22333333333333336, 0.22333333333333336, 0.20809523809523806, 0.17571428571428568, 0.19238095238095237, 0.18666666666666668, 0.21619047619047624, 0.18476190476190477, 0.16904761904761903, 0.23523809523809525, 0.20571428571428568, 0.19380952380952382, 0.23666666666666666, 0.2176190476190476, 0.2061904761904762, 0.22190476190476197, 0.19904761904761903, 0.18476190476190477, 0.18000000000000002, 0.21047619047619048, 0.17476190476190476, 0.2061904761904762, 0.23761904761904767, 0.19428571428571428, 0.2123809523809524, 0.2585714285714286, 0.26857142857142857, 0.21476190476190476, 0.21333333333333332, 0.2061904761904762, 0.23285714285714287, 0.1980952380952381, 0.22952380952380955, 0.20809523809523806, 0.20523809523809522, 0.2, 0.19666666666666666, 0.17809523809523808, 0.2061904761904762, 0.21476190476190476, 0.17571428571428568, 0.20571428571428574, 0.20380952380952383, 0.21714285714285717, 0.20428571428571432, 0.2228571428571428, 0.23238095238095238, 0.23761904761904767, 0.18095238095238095, 0.20904761904761904, 0.27, 0.1761904761904762, 0.2028571428571429, 0.23142857142857146, 0.1857142857142857, 0.17238095238095236, 0.21619047619047618, 0.20238095238095238, 0.22285714285714284, 0.27, 0.2228571428571429, 0.19952380952380955, 0.20190476190476192, 0.24095238095238095, 0.23142857142857146, 0.23047619047619047, 0.19571428571428567, 0.21333333333333332, 0.24142857142857144, 0.23666666666666666, 0.20571428571428574, 0.21714285714285717, 0.21761904761904763, 0.20809523809523806, 0.16666666666666666, 0.19380952380952382, 0.20428571428571432, 0.18857142857142856, 0.23190476190476192, 0.21714285714285714, 0.21476190476190476, 0.2080952380952381, 0.2314285714285714, 0.21857142857142856, 0.22, 0.2138095238095238, 0.17476190476190476, 0.19952380952380955, 0.15523809523809523, 0.21666666666666665, 0.1895238095238095, 0.17476190476190476, 0.23095238095238102, 0.19714285714285712, 0.23380952380952383, 0.23047619047619047, 0.19476190476190475, 0.15523809523809523, 0.21666666666666665, 0.17476190476190476, 0.22333333333333336, 0.19476190476190475, 0.20857142857142857, 0.21666666666666665, 0.22, 0.23761904761904767, 0.18666666666666668, 0.21047619047619048, 0.20809523809523806, 0.1814285714285714, 0.21666666666666665, 0.2252380952380952, 0.20904761904761904, 0.21333333333333332, 0.17238095238095236, 0.16999999999999998, 0.2333333333333333, 0.2119047619047619, 0.2028571428571428, 0.2080952380952381, 0.20380952380952377, 0.22999999999999995, 0.2147619047619048, 0.17476190476190476, 0.21333333333333332, 0.21380952380952378, 0.2247619047619048, 0.17714285714285713, 0.20380952380952377, 0.19476190476190475, 0.17476190476190476, 0.16857142857142857, 0.1857142857142857, 0.21619047619047624, 0.22428571428571428, 0.23238095238095238, 0.21047619047619048, 0.17285714285714285, 0.21428571428571425, 0.17476190476190476, 0.2061904761904762, 0.19476190476190475, 0.18571428571428572, 0.21238095238095242, 0.19142857142857145, 0.22523809523809527, 0.1780952380952381, 0.19714285714285718, 0.17476190476190476, 0.22428571428571428, 0.2123809523809524, 0.21047619047619048, 0.20952380952380953, 0.23238095238095233, 0.18000000000000002, 0.20761904761904765, 0.27047619047619054, 0.21047619047619048, 0.23380952380952383, 0.21333333333333332, 0.22095238095238093, 0.17476190476190476, 0.2061904761904762, 0.17476190476190476, 0.19666666666666666, 0.2042857142857143, 0.19714285714285718, 0.2061904761904762, 0.21666666666666665, 0.2547619047619048, 0.17571428571428568, 0.21428571428571427, 0.21000000000000002, 0.20904761904761904, 0.2061904761904762, 0.2047619047619047, 0.2176190476190476, 0.259047619047619, 0.20999999999999996, 0.20904761904761904, 0.1980952380952381, 0.17285714285714285, 0.23238095238095238, 0.23714285714285713, 0.21047619047619048, 0.23238095238095238]
#optimal_xs: [63.35683146876411, 61.22056031713039, 19.087195270919167, 13.398084245829654, 23.316066817220168, 43.00271851162911, 61.09437929178371, 13.480371652172638, 67.7057656201406, 41.763213273100526, 74.39361414338924, 41.763213273100526, 36.076901134547775, 57.47666987679189, 81.46463325479621, 81.93950453893314, 54.13500410394892, 58.683675587966775, 37.27545325803041, 81.73649335167903, 21.199990321562876, 82.47452158470857, 16.444067087544912, 41.763213273100526, 36.37413047414657, 24.716109453029556, 55.25674176393824, 55.25674176393824, 22.027569735577273, 82.47469468908434, 21.98863390267828, 82.47467264525656, 45.93381399211005, 42.94326590547159, 1.3257983867443535, 12.686137835150019, 5.940625994597504, 84.26005395976179, 72.43671492704387, 41.763213273100526, 38.526365619517144, 12.591676259490692, 27.00854645444577, 12.528694933013513, 75.33963376113701, 41.763213273100526, 69.53171124668268, 42.09003847294665, 5.238160118308914, 81.79129633632186, 58.21185406842148, 58.21185406842148, 53.7893251309133, 82.84734740709493, 80.36198645547375, 82.86288024812707, 65.12300641465981, 82.21196038152046, 87.58712895805172, 83.67132205300656, 61.76830809383822, 59.62734806852822, 25.83867327165838, 12.58052824805414, 8.04454917859371, 12.481913991749995, 77.42470354616424, 13.432126531048393, 67.53875616921783, 41.763213273100526, 64.9963246452926, 82.14302735958951, 36.7784305429761, 55.29802960603488, 33.522307314072805, 12.716607057742845, 89.91996440617936, 84.28321591374028, 46.46738780179444, 78.20415349282865, 19.903496073783803, 58.32505432284955, 70.0713449155589, 43.00271851162911, 9.123852147343982, 12.538783396809201, 64.82969599828212, 82.06999200669851, 22.218237502357063, 82.4747952711805, 4.1195698018954925, 82.55960907956384, 62.66413531702479, 60.52140740597357, 24.262243413544514, 56.23667999781195, 54.47696766014032, 42.94054023308625, 71.21366810387697, 42.50874792405802, 66.79676038421375, 13.096637785417645, 12.454871015466507, 12.454871015466507, 14.536930663736701, 13.224922286170605, 8.344228595073018, 82.57845453603726, 74.08278479007846, 41.763213273100526, 24.985524912464626, 12.224249535721096, 78.60344687737201, 12.941922196854902, 26.7399960853918, 12.502410676915563, 13.978056212895394, 12.666016618049682, 19.890438188899115, 55.6678739042274, 89.01702825475913, 41.80169202067649, 47.53418726414979, 31.74931385041377, 60.650926781605, 13.253744720072651, 7.24547402386664, 82.08115351024234, 47.43782640990336, 27.878085341515664, 81.84736502923307, 13.397440974542128, 60.42832446170029, 60.42832446170029, 81.69673647296301, 83.92494342687066, 58.7645163788691, 12.910789653985923, 83.7517854562212, 83.7517854562212, 18.81817906521025, 13.168170055799607, 66.62976821482806, 13.096594164349293, 15.597312415443277, 56.7867343032413, 47.45045961182791, 27.564904630693746, 8.326519971453752, 42.48126942709283, 78.94386260993956, 12.96701291857198, 24.81055470656503, 59.82592972281601, 19.760583346794956, 55.91266137126208, 23.46353235741877, 83.65407551465502, 50.24153982591774, 58.70737071992057, 11.004732407403822, 13.07077122114195, 9.329507331345768, 12.726479220835113, 48.09589006816926, 78.13508775178687, 53.395794545159845, 82.83211137702013, 25.285687858642014, 12.297687199513868, 26.736388179613556, 12.496963828273412, 62.09971992762513, 60.25089766516245, 45.940949790061474, 42.94054773729765, 2.2160373350477123, 83.06485259818822, 58.21724800140206, 58.21724800140206, 12.701475275451958, 12.701475275451958, 45.32371152258046, 12.105252839499542, 87.98723105652509, 84.97587133860043, 31.54634222079376, 42.3963122314848, 65.7605822806774, 13.096659222965286, 87.11283605828531, 12.961471424989053, 89.48058870308209, 84.88308912212506, 15.079540320416383, 55.20735470726558, 3.6383070754066047, 83.39116567531684, 41.69304908194008, 41.69304908194008, 16.826610889083966, 41.763213273100526, 78.6423458965109, 13.036615814522854, 61.448676836711655, 13.484430107436024, 32.263260724053964, 12.290625847214912, 72.40040328503318, 41.763213273100526, 18.51148009589881, 41.763213273100526, 47.01456820631451, 42.69945018154278, 59.00001382061964, 13.129914188434173, 63.08176332882141, 61.30625753440871, 50.16526842163006, 58.67190165512891, 65.33714789181975, 13.096615435194956, 82.2165432717154, 13.525204877184619, 24.0661359959592, 56.72596029265612, 18.07482930495372, 41.763213273100526, 21.560867330701765, 82.47439813585459, 44.893752842419914, 13.287285780755525, 13.49329377514054, 13.49329377514054, 7.820325219772394, 83.41264751919185, 81.51880853560905, 83.71985493501695, 36.659282951586306, 55.237483724370136, 71.79258096268764, 41.763213273100526, 4.841242807637966, 82.28510638912184, 45.99025129173146, 42.98815194123356, 2.0220334230229664, 43.088567951654476, 7.818070930649722, 83.4238116751863, 44.477714071368325, 79.09462858537452, 34.97599444273131, 55.63438628683936, 9.252259913757356, 12.65306851913055, 84.31412030459414, 84.31412030459414, 84.58462333062143, 84.58462333062143, 20.635379940377803, 42.575438317315054, 30.461955607064066, 84.13350570976725, 24.878777530225726, 56.445993321243186, 14.480821193893993, 58.53619560138499, 43.48875347545569, 42.17672699696321, 19.56161336604775, 42.72520182309181, 88.23269588422481, 12.51411722679422, 17.907500102376126, 41.763213273100526, 9.447271510508754, 12.842282659839302, 43.9233524307032, 42.6113434165234, 64.3531713499973, 57.806814565100076, 76.4523324786325, 83.55039018779433, 39.61640313406843, 12.157937790078796, 59.6809896855, 12.85420169192656, 67.658034507677, 41.763213273100526, 76.65476029024964, 82.90802262642676, 58.83086202768425, 12.903345952671456, 39.89467462720541, 12.698976042177529, 24.291441087352524, 83.67122342539624, 69.11881810686235, 41.74585004730582, 5.883576403502739, 82.25130369519489, 83.56426724213718, 13.245162388190815, 44.12637695694128, 42.8142953142258, 67.19134208968917, 41.763213273100526, 64.77142688140594, 82.03895227817789, 65.40248266662157, 13.096636788210223, 45.45223729771958, 12.305738823954652, 27.33441167430518, 12.476618040087295, 23.557295278921973, 43.00271851162911, 33.41006547673765, 82.67562773410779, 12.016374213758032, 12.016374213758032, 28.82529877475074, 13.566694822606234, 18.413317443548532, 41.763213273100526, 24.70655498661049, 83.67825385910129, 89.89426942921415, 84.28798948025819, 5.897825175539921, 82.25130369519489, 27.378324370572535, 12.49914940447546, 49.456501754848155, 82.40929850895125, 7.805125777056842, 12.276580642986438, 77.22504665617434, 13.382371632709543, 20.34757976063894, 31.630452385467642, 76.9775481251985, 13.618508852680707, 48.00362007907409, 41.43283753928452, 10.285599016340862, 12.400369099058603, 6.806273360898331, 22.971520395197416, 16.663987316301466, 41.763213273100526, 14.415023269024918, 13.10301226936594, 18.558402368265263, 41.763213273100526, 13.139821979078448, 13.139821979078448, 50.556149824281796, 13.4770735231262, 23.414181693977255, 43.00271851162911, 64.62370305687277, 83.2016214129927, 33.576791731993204, 12.290625847214912, 33.376680272581105, 31.036320366057378, 75.23886052825648, 41.763213273100526, 18.76527701036368, 13.125642938263782, 71.01877675416519, 42.87886656508579, 7.1005842297852, 82.08077918392019, 62.868055415970794, 60.72719921247458, 11.644198719577057, 42.75135052460612, 87.29354459672629, 84.20170253994598, 36.68987410216149, 57.506554438504644, 18.995207710979514, 13.31910520571968, 36.24206531149494, 55.670039467416395, 36.51488161660021, 8.529112417118693, 83.46156358193807, 13.266084686478042, 69.04912739633383, 41.74585004730582, 87.95376157453673, 85.2316665202321, 77.62503497122097, 13.346874466846234, 35.73412779894919, 78.06057861359146]
#optimal_fxs: [0.56, 0.71, 0.53, 0.76, 0.7, 0.71, 0.71, 0.76, 0.31, 0.71, 0.3, 0.71, 0.65, 0.71, 0.63, 0.71, 0.5, 0.71, 0.65, 0.71, 0.27, 0.71, 0.63, 0.71, 0.65, 0.7, 0.71, 0.71, 0.27, 0.71, 0.27, 0.71, 0.53, 0.71, 0.59, 0.76, 0.59, 0.71, 0.3, 0.71, 0.44, 0.76, 0.65, 0.76, 0.3, 0.71, 0.53, 0.71, 0.59, 0.71, 0.71, 0.71, 0.5, 0.71, 0.63, 0.71, 0.44, 0.71, 0.37, 0.71, 0.56, 0.71, 0.5, 0.76, 0.47, 0.76, 0.67, 0.76, 0.31, 0.71, 0.44, 0.71, 0.65, 0.71, 0.63, 0.76, 0.45, 0.71, 0.53, 0.67, 0.35, 0.71, 0.53, 0.71, 0.67, 0.76, 0.44, 0.71, 0.27, 0.71, 0.56, 0.71, 0.56, 0.71, 0.7, 0.71, 0.5, 0.71, 0.53, 0.71, 0.22, 0.76, 0.76, 0.76, 0.47, 0.76, 0.67, 0.71, 0.3, 0.71, 0.5, 0.76, 0.67, 0.76, 0.65, 0.76, 0.47, 0.76, 0.35, 0.71, 0.37, 0.71, 0.41, 0.65, 0.71, 0.76, 0.47, 0.71, 0.41, 0.65, 0.71, 0.76, 0.71, 0.71, 0.63, 0.71, 0.71, 0.76, 0.71, 0.71, 0.53, 0.76, 0.22, 0.76, 0.47, 0.71, 0.41, 0.65, 0.67, 0.71, 0.67, 0.76, 0.5, 0.71, 0.35, 0.71, 0.7, 0.71, 0.29, 0.71, 0.59, 0.76, 0.67, 0.76, 0.41, 0.67, 0.5, 0.71, 0.5, 0.76, 0.65, 0.76, 0.56, 0.71, 0.53, 0.71, 0.59, 0.71, 0.71, 0.71, 0.76, 0.76, 0.53, 0.76, 0.37, 0.71, 0.65, 0.71, 0.22, 0.76, 0.56, 0.76, 0.45, 0.71, 0.47, 0.71, 0.56, 0.71, 0.71, 0.71, 0.63, 0.71, 0.67, 0.76, 0.71, 0.76, 0.63, 0.76, 0.3, 0.71, 0.53, 0.71, 0.41, 0.71, 0.71, 0.76, 0.56, 0.71, 0.29, 0.71, 0.22, 0.76, 0.71, 0.76, 0.7, 0.71, 0.53, 0.71, 0.27, 0.71, 0.44, 0.76, 0.76, 0.76, 0.47, 0.71, 0.63, 0.71, 0.65, 0.71, 0.3, 0.71, 0.59, 0.71, 0.53, 0.71, 0.59, 0.71, 0.47, 0.71, 0.44, 0.67, 0.35, 0.71, 0.67, 0.76, 0.71, 0.71, 0.71, 0.71, 0.35, 0.71, 0.65, 0.71, 0.5, 0.71, 0.47, 0.71, 0.44, 0.71, 0.35, 0.71, 0.37, 0.76, 0.53, 0.71, 0.67, 0.76, 0.44, 0.71, 0.44, 0.71, 0.67, 0.71, 0.44, 0.76, 0.71, 0.76, 0.31, 0.71, 0.67, 0.71, 0.71, 0.76, 0.44, 0.76, 0.7, 0.71, 0.53, 0.71, 0.59, 0.71, 0.71, 0.76, 0.44, 0.71, 0.31, 0.71, 0.44, 0.71, 0.22, 0.76, 0.53, 0.76, 0.65, 0.76, 0.7, 0.71, 0.63, 0.71, 0.76, 0.76, 0.61, 0.76, 0.53, 0.71, 0.7, 0.71, 0.45, 0.71, 0.59, 0.71, 0.65, 0.76, 0.29, 0.71, 0.47, 0.76, 0.67, 0.76, 0.35, 0.65, 0.67, 0.76, 0.41, 0.71, 0.59, 0.76, 0.47, 0.7, 0.63, 0.71, 0.47, 0.76, 0.53, 0.71, 0.76, 0.76, 0.5, 0.76, 0.7, 0.71, 0.44, 0.71, 0.63, 0.76, 0.63, 0.65, 0.3, 0.71, 0.53, 0.76, 0.53, 0.71, 0.47, 0.71, 0.56, 0.71, 0.59, 0.71, 0.37, 0.71, 0.65, 0.71, 0.53, 0.76, 0.65, 0.71, 0.65, 0.67, 0.71, 0.76, 0.53, 0.71, 0.37, 0.71, 0.67, 0.76, 0.35, 0.67]
#times: [2.724883794784546, 2.864835739135742, 2.5796010494232178, 2.58709454536438, 2.4300804138183594, 2.586618185043335, 2.5065739154815674, 2.485316038131714, 2.2442638874053955, 3.055283784866333, 2.6369998455047607, 2.375900983810425, 2.443877935409546, 1.8181302547454834, 2.4194977283477783, 2.772921085357666, 2.624035120010376, 2.674121618270874, 2.613210678100586, 2.2771997451782227, 2.760392904281616, 2.6509904861450195, 2.42415189743042, 2.654419422149658, 2.4506070613861084, 2.692805528640747, 2.938887357711792, 2.8705785274505615, 2.588358163833618, 2.468026876449585, 2.4488420486450195, 2.2227461338043213, 2.5575737953186035, 2.693037748336792, 2.5526134967803955, 2.4553303718566895, 2.4480719566345215, 3.1929078102111816, 2.4007959365844727, 2.2901110649108887, 2.4854400157928467, 2.408463954925537, 2.685269832611084, 2.608858585357666, 1.9455626010894775, 2.4777395725250244, 2.349062442779541, 2.261720895767212, 2.4512994289398193, 2.1035163402557373, 2.7795052528381348, 2.6106269359588623, 2.337552309036255, 2.7170698642730713, 2.3755314350128174, 2.7020108699798584, 2.5999631881713867, 2.385317087173462, 2.4937894344329834, 2.595587968826294, 2.6762146949768066, 2.715742826461792, 2.561811685562134, 2.7010412216186523, 2.4118993282318115, 2.719588279724121, 2.8657326698303223, 2.5042572021484375, 1.6364502906799316, 2.535128593444824, 3.0555169582366943, 2.880889654159546, 2.593975305557251, 2.0633492469787598, 2.8226566314697266, 2.6181395053863525, 2.4765684604644775, 2.0316078662872314, 2.5122833251953125, 2.5825419425964355, 2.9466326236724854, 2.994699001312256, 2.541088819503784, 2.764866352081299, 2.758043050765991, 2.5910403728485107, 2.6277358531951904, 2.1264758110046387, 1.9765331745147705, 2.9162023067474365, 2.542947292327881, 2.3333933353424072, 2.083474636077881, 2.711177349090576, 2.691605806350708, 2.388867139816284, 2.6000587940216064, 2.787477731704712, 2.757086753845215, 2.5523016452789307, 1.8570997714996338, 2.5819640159606934, 2.1192588806152344, 2.3385181427001953, 2.498063087463379, 2.5904502868652344, 2.140346050262451, 2.5592973232269287, 2.0817534923553467, 2.5476489067077637, 2.6608190536499023, 2.4689977169036865, 2.336263418197632, 1.5673604011535645, 2.312709093093872, 2.871222972869873, 2.6912896633148193, 1.8397908210754395, 2.6803154945373535, 2.6214447021484375, 2.5405383110046387, 2.8489372730255127, 2.3967204093933105, 2.8777687549591064, 2.80643892288208, 2.2882509231567383, 2.904322862625122, 2.916991710662842, 2.7812001705169678, 2.5649569034576416, 2.792938470840454, 2.888195276260376, 2.6890578269958496, 2.9928243160247803, 2.744035482406616, 2.8082454204559326, 2.578376054763794, 2.5391135215759277, 2.679945945739746, 2.5139598846435547, 2.655855417251587, 2.225231885910034, 2.490122079849243, 2.579179286956787, 2.219473123550415, 2.2864952087402344, 2.8979785442352295, 2.9033329486846924, 2.6604928970336914, 2.695919990539551, 2.9497060775756836, 2.887723922729492, 2.6842617988586426, 2.5856776237487793, 2.918503522872925, 2.668121337890625, 2.6635806560516357, 2.388406276702881, 1.8460016250610352, 2.6085751056671143, 2.646244525909424, 2.5747873783111572, 2.5678799152374268, 2.3949289321899414, 2.265456199645996, 1.9984104633331299, 2.3116514682769775, 2.355497121810913, 1.6557037830352783, 2.206979990005493, 2.5527708530426025, 1.8023135662078857, 2.211475372314453, 2.5377707481384277, 2.5707361698150635, 2.4530622959136963, 2.411627769470215, 2.4906365871429443, 2.185889482498169, 2.2813491821289062, 2.6802096366882324, 2.6617839336395264, 2.6023261547088623, 2.687546968460083, 2.0351648330688477, 2.856228828430176, 2.4400110244750977, 2.3965959548950195, 2.5144195556640625, 2.613229990005493, 2.4764368534088135, 2.0940053462982178, 1.8554728031158447, 1.9298639297485352, 2.5925092697143555, 2.2146639823913574, 2.5555665493011475, 2.5852723121643066, 2.611496925354004, 2.3511266708374023]



#Sampling count: 11
#Average time taken: 1.0651256918907166
#optimal_for_this_maxiter_xs: [23.725417681539756, 16.523545904355863, 24.300744799655178, 42.54162911083237, 81.89722070583656, 37.03717048675196, 83.55039018779433, 60.487397914746175, 76.8002254077314, 41.763213273100526, 41.42208710216682, 82.5362379734378, 41.763213273100526, 85.09309858885054, 41.763213273100526, 42.14810316264899, 13.096594164349293, 12.788343648774754, 23.176977888973774, 76.79928399839083, 41.763213273100526, 12.716873739546273, 13.003693507330846, 37.60786762630525, 41.936352970796555, 60.89449501542673, 11.487472836612081, 84.72434298603011, 22.972613780245794, 12.653467110745211, 81.75245455736433, 2.701978729910116, 85.21474020841188, 41.763213273100526, 23.718908184238312, 83.55039018779433, 13.096616430523682, 84.10728434036513, 9.48086257576281, 37.303363518913365, 42.92816684279827, 13.253744720072651, 16.523545904355863, 37.433532944115946, 42.50874792405802, 2.6950106889477166, 76.33697711839869, 24.478627056506756, 60.19815990041261, 30.923746799404473, 83.55039018779433, 43.00271851162911, 12.192193866412293, 8.643629239503332, 12.348009728151123, 41.93960454168972, 11.974466294560202, 36.01035419855643, 43.00271851162911, 60.2056331181877, 83.6841866104364, 31.303731635983485, 13.096660222101544, 23.138412092936786, 31.516296743547688, 83.86883564835613, 41.78569295355097, 82.97157889177413, 42.95812456149248, 41.78646614281808, 41.799478285606845, 13.096615435194956, 37.49244207522799, 31.266085701215662, 60.0461218310299, 8.652556043838182, 43.00271851162911, 60.07500620530772, 41.763213273100526, 41.763213273100526, 43.09898751787821, 12.264646890381913, 42.84465398800176, 31.533171410301698, 82.99138857663019, 43.13209021009244, 55.61420985040574, 55.41506160719883, 30.877646902428687, 8.523251958863053, 83.55146405266298, 55.26946122476258, 42.67623220325729, 60.19815990041261, 35.861857849577284, 8.644056166139348, 13.09668074364731, 12.880542887519919, 16.63581929673667, 13.096637785417645]
#optimal_for_this_maxiter_fxs: [0.7, 0.63, 0.7, 0.71, 0.71, 0.65, 0.71, 0.71, 0.67, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.76, 0.76, 0.7, 0.67, 0.71, 0.76, 0.76, 0.65, 0.71, 0.71, 0.59, 0.71, 0.7, 0.76, 0.71, 0.59, 0.71, 0.71, 0.7, 0.71, 0.76, 0.71, 0.67, 0.65, 0.71, 0.76, 0.63, 0.65, 0.71, 0.59, 0.67, 0.7, 0.71, 0.65, 0.71, 0.71, 0.76, 0.67, 0.76, 0.71, 0.76, 0.65, 0.71, 0.71, 0.71, 0.65, 0.76, 0.7, 0.65, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.76, 0.65, 0.65, 0.71, 0.67, 0.71, 0.71, 0.71, 0.71, 0.71, 0.76, 0.71, 0.65, 0.71, 0.71, 0.71, 0.71, 0.65, 0.67, 0.71, 0.71, 0.71, 0.71, 0.65, 0.67, 0.76, 0.76, 0.63, 0.76]
#avg_cum_regrets_for_this_maxiter: 0.23428181818181815
#avg_optimal_for_this_maxiter_fxs: 0.7003000000000001
#------------------------------------------------------------------
#AVG Found optima With 100 repeats and 11 Iterations: y=0.7003000000000001
#------------------------------------------------------------------
#Global optima: x=-32.142857 y=0.76
#INFO: Appending 0.7003000000000001 to average_optimal_fxs
#CURRENT: maxiter_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#CURRENT: average_optimal_fxs (index is iteration): [0.5412, 0.574, 0.5882999999999999, 0.6371999999999999, 0.6477, 0.6453, 0.6657, 0.6795999999999996, 0.6890999999999998, 0.6909000000000001, 0.7003000000000001]
#11 11
#
#Process finished with exit code 0