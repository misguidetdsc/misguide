#######################################################################
###################### Importing libraries ############################
#######################################################################
import numpy as np
import pandas as pd
import time
import math
import gurobipy as gp
import time as tm
import json
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
# Disable gurobi warnings
gp.setParam("OutputFlag", 0)


class MISGUIDE:
    
    def __init__(self, bus_system, number_days_load_data, nominal_frequency, agc_cycle, generator_sampling_time, agc_sampling_time, uf_threshold, of_threshold):
        
        # Initializing class variables
        self.bus_system              = bus_system             # Number of IEEE bus system (e.g., 39, 68, 118)  
        self.number_days_load_data   = number_days_load_data  # Number of days used to generate the load dataframe
        self.nominal_frequency       = nominal_frequency
        self.agc_cycle               = agc_cycle
        self.generator_sampling_time = generator_sampling_time
        self.agc_sampling_time       = agc_sampling_time
        self.uf_threshold            = uf_threshold
        self.of_threshold            = of_threshold

    def initialize_files(self, prefix_filename_dataset, prefix_filename_bus_susceptance_matrix, prefix_filename_bus_parameters, prefix_filename_generator_parameters):
        
        # Populating variables with filename
        self.dataframe              = pd.read_csv(prefix_filename_dataset + str(self.number_days_load_data) + '-Days.csv')
        self.bus_susceptance_matrix = pd.read_csv(prefix_filename_bus_susceptance_matrix + str(self.bus_system) + '-Bus.csv', header = None)
        self.bus_parameters         = pd.read_csv(prefix_filename_bus_parameters + str(self.bus_system) + '-Bus.csv')
        self.generator_parameters   = pd.read_csv(prefix_filename_generator_parameters + str(self.bus_system) + '-Bus.csv')
        
    def initialize_bus_parameters(self, generator_with_governors):
        
        self.slack_buses     = self.bus_parameters[self.bus_parameters['Bus Type'] == 1]['Bus No.'].values    # slack buses
        self.pv_buses        = self.bus_parameters[self.bus_parameters['Bus Type'] == 2]['Bus No.'].values    # slack buses
        self.pq_buses        = self.bus_parameters[self.bus_parameters['Bus Type'] == 3]['Bus No.'].values    # slack buses
        self.delta_ref       = np.append(None, self.bus_parameters['Bus Phase Angle'].values)
        self.pg_ref          = np.append(None, self.bus_parameters['Active Power Generation Output'].values / 2) # scaling factor 0.5 is multiplied to balance the initial load measurements with the generator reference
        
        self.generator_buses             = np.append(self.pv_buses, self.slack_buses)
        self.generator_with_governors    = generator_with_governors
        self.generator_with_no_governors = np.setdiff1d(self.generator_buses, self.generator_with_governors)  # generators with no governors
        self.all_buses                   = np.array([i for i in range(1, self.bus_system + 1)])   # set of all bus indexes        
        
    def initialize_generator_parameters(self):
        
        self.map_bus_generators       = np.full(self.bus_system + 1, None, dtype = object)
        self.map_generator_buses      = np.full(len(self.generator_parameters) + 1, None, dtype = object)
        self.generator_droops         = np.full(len(self.generator_parameters) + 1, None, dtype = object)
        self.generator_time_constants = np.full(len(self.generator_parameters) + 1, None, dtype = object)
        self.generator_inertia        = np.full(len(self.generator_parameters) + 1, None, dtype = object)
                                       
        
        for i in range(len(self.generator_parameters)):
            self.map_generator_buses[i + 1]                                          = self.generator_parameters['Associated Bus'][i]
            self.map_bus_generators[self.generator_parameters['Associated Bus'][i]]  = i + 1
            self.generator_droops[i + 1]                                             = self.generator_parameters['Droop Co-efficient'][i]
            self.generator_time_constants[i + 1]                                     = self.generator_parameters['Time Constant'][i]
            self.generator_inertia[i + 1]                                            = self.generator_parameters['Governor Inertia'][i]    
    
    def bdd_load_threshold(self, load_interval):   
        
        num_benign_loads  = math.ceil(len(self.dataframe) / load_interval)
        
        self.benign_loads = np.zeros((self.bus_system + 1, num_benign_loads))
        
        for b in range(1, self.bus_system + 1):
            for t in range(num_benign_loads):
                self.benign_loads[b][t] = self.dataframe['Load Measurements in Bus' + str(b) + " (PU)"][t * load_interval]
        
        self.load_thresholds_pos = [None]
        self.load_thresholds_neg = [None]
        self.attackable_buses = []
        
        for b in range(1, self.bus_system + 1):
            max_dev = 0
            min_dev = 0
            for t in range(1, num_benign_loads):
                dev = self.benign_loads[b][t] - self.benign_loads[b][t - 1]
                if dev >= 0:
                    max_dev = max(max_dev, dev)
                else:
                    min_dev = min(min_dev, dev)
                    
            self.load_thresholds_pos.append(max_dev)
            self.load_thresholds_neg.append(min_dev)
            
            if max_dev != 0 or min_dev != 0:
                self.attackable_buses.append(b)
        self.no_load_generator_buses     = np.setdiff1d(self.generator_buses, np.array(self.attackable_buses))  # generators with no loads
        
    
    def adm_load_threshold(self):
        
        self.all_hulls = []
        self.all_vertices = []
        
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:            
                dataframe = pd.DataFrame()
                previous_measurements  = []
                current_measurements   = []
                deviation_measurements = []
                
                for t in range(1, len(self.benign_loads[b]) ):
                    previous_measurements.append(self.benign_loads[b][t - 1])
                    deviation_measurements.append(self.benign_loads[b][t] - self.benign_loads[b][t - 1])
                
                dataframe['Previous Load Measurements (P.U.)'] = previous_measurements
                dataframe['Deviation Measurements (P.U.)']     = deviation_measurements
                
                clustering = DBSCAN(eps = 10, min_samples = 5).fit(dataframe.values)
                labels = clustering.labels_
                num_clusters = len(set(labels) - {-1})
                
                num_noise_points = sum(labels == -1)
                
                clusters = []
                for c in range(num_clusters):
                    cluster_points = []
                    for l in range(len(labels)):
                        if labels[l] == c:
                            cluster_points.append(l)
                    cluster_points = np.array(cluster_points)
                    
                    clusters.append(cluster_points)
                    
                final_clusters = []
                hulls    = []
                vertices = []
                for c in range(len(clusters)):
                    points = []
                    for p in clusters[c]:
                        points.append(dataframe.values[p][:2].tolist())
                    points = np.array(points)
                 
                    hull = ConvexHull(points)
                    hulls.append(hull.equations)
                    
                    vertices.append(np.append(points[hull.vertices], points[hull.vertices[0]].reshape(1, -1), axis = 0))
                    
                self.all_hulls.append(hulls)
                self.all_vertices.append(vertices)
            
            else:
                self.all_hulls.append(None)
                self.all_vertices.append(None)

    def visualize(self, plt_title, gp_mat, scale, mat_x_label, mat_y_label, mat_bus_indexes, mat_time_range, mat_legend_labels, mat_legend_location, plt_range, output_filename):
    
        all_vals = []
        for b in mat_bus_indexes:
            vals = []
            for t in range(mat_time_range):
                vals.append(gp_mat[b][t] * scale)
    
            all_vals.append(vals)
    
        all_vals = [list(x) for x in zip(*all_vals)]
        all_vals = np.array(all_vals)
    
        
        plt.figure(figsize = (9, 6))
        plt.grid()
        plt.title(plt_title)
        plt.xlabel(mat_x_label, fontsize = "25")
        plt.ylabel(mat_y_label, fontsize = "25")
        plt.yticks(fontsize="22")
        plt.xticks(fontsize="22")
        plt.plot(all_vals)
    
        if plt_range != "":
            plt.ylim(plt_range)
        plt.legend(labels = mat_legend_labels, loc = mat_legend_location, bbox_to_anchor=(1.0, 1.0), ncol = 4, fontsize = "18")

    def benign_system_analysis(self, number_timeslots, variable_load_timeslots, filename_to_save_output):
               
        start_time = time.time()
        self.number_timeslots        = number_timeslots
        self.number_agc_steps        = int(self.number_timeslots / self.agc_cycle)
        
        model = gp.Model()
        
        # Initializing AGC measurement variables
        gp_agc_pg    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='gp_agc_pg_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pl    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pl_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pr    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pr_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pm    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pm_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_delta = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_delta_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_f     = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_f_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]

        # Initializing system measurement variables
        gp_gen_pg    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='gp_gen_pg_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_pl    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_pl_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_pm    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_pm_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_delta = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_delta_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_f     = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_f_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_sys_f     = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'gp_sys_f')
        
        gp_load      = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_load_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        
        #######################################################################
        ########################## System dynamics ############################
        #######################################################################
        
        # [BUS] Benign load measurements
        for t in range(self.number_timeslots):
            for b in range(1, self.bus_system + 1):
                if t < (variable_load_timeslots):
                    model.addConstr(gp_load[b][t] == self.benign_loads[b][int(t / self.agc_cycle)])
                else:
                    model.addConstr(gp_load[b][t] == self.benign_loads[b][int(variable_load_timeslots / self.agc_cycle)])
            
        # [BUS] Actual power flow 
        for t in range(self.number_timeslots):
            for b_1 in range(1, self.bus_system + 1):
                to_add = 0
                for b_2 in range(1, self.bus_system + 1):
                    to_add += self.bus_susceptance_matrix[b_1 - 1][b_2 - 1] * (gp_gen_delta[b_1][t] - gp_gen_delta[b_2][t])  
                
                model.addConstr(gp_gen_pg[b_1][t] - gp_load[b_1][t] == to_add)
                
        # [GEN] active power output for pg buses in the initial timeslot
        for b in self.pv_buses:
            model.addConstr(gp_gen_pg[b][0] == self.pg_ref[b])
            
        # [GEN] active power output threshold for pg buses
        for b in self.pv_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pg[b][t] >= 0)  
                
        # [GEN] active power output for for pq buses      
        for b in self.pq_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pg[b][t] == 0)
                
        # [GEN] swing equation constraints for generators with no governors
        for b in self.generator_with_no_governors:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pm[b][t] == gp_gen_pg[b][0] * 0.1)
        
        # [GEN] initial timeslot swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            model.addConstr(gp_gen_pm[b][0] == gp_agc_pr[b][0] * 20)
        
        # [GEN] other timeslots swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                model.addConstr(gp_gen_pm[b][t + 1] == gp_gen_pm[b][t] + self.generator_sampling_time / self.generator_time_constants[self.map_bus_generators[b]] * ((gp_agc_pr[b][t + 1] - (gp_gen_f[b][t + 1] - gp_gen_f[b][0])) * 20 - gp_gen_pm[b][t + 1]))
        
        # [BUS] nominal frequency constraint
        for b in range(1, self.bus_system + 1):
            model.addConstr(gp_gen_f[b][0] == 1)
        
        # [BUS] frequency constraints for generators with governors
        for t in range(self.number_timeslots - 1):
            for b in self.generator_buses:
                model.addConstr(gp_gen_f[b][t + 1] == gp_gen_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_gen_pm[b][t + 1] - 0.1 * gp_gen_pg[b][t + 1]))
        
        # [GEN] rotor angle initialization for slack buses (reference bus initialization)
        for b in self.slack_buses:
            model.addConstr(gp_gen_delta[b][0] == self.delta_ref[b] * (math.pi / 180))
                
        # [BUS] phase angle constraints
        for t in range(self.number_timeslots - 1):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_gen_delta[b][t + 1] == gp_gen_delta[b][t] + 2 * math.pi * self.generator_sampling_time * (gp_gen_f[b][t + 1] - gp_gen_f[b][0]))


        #######################################################################
        ########### Automatic Generation Control Controller ###################
        #######################################################################

        # [AGC] Power flow analysis
        for t in range(self.number_timeslots):
            for b_1 in range(1, self.bus_system + 1):
                to_add = 0
                for b_2 in range(1, self.bus_system + 1):
                    to_add += self.bus_susceptance_matrix[b_1 - 1][b_2 - 1] * (gp_agc_delta[b_1][t] - gp_agc_delta[b_2][t])  
                
                model.addConstr(gp_agc_pg[b_1][t] - gp_load[b_1][t] == to_add)
                
        
        # [AGC] active power output for pv bus generators
        for b in self.pv_buses:
            model.addConstr(gp_agc_pg[b][0] == self.pg_ref[b] )    
    
        # [AGC] active power output for for pq buses      
        for b in self.pq_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_agc_pg[b][t] == 0)  
                
        # [AGC] swing equation constraints for generators with no governors
        for b in self.generator_with_no_governors:
            for t in range(self.number_timeslots):
                model.addConstr(gp_agc_pm[b][t] == gp_agc_pg[b][0] * 0.1)
        
        # [AGC] initial timeslot swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            model.addConstr(gp_agc_pm[b][0] == gp_agc_pr[b][0] * 20)
        
        # [AGC] other timeslots swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                model.addConstr(gp_agc_pm[b][t + 1] == gp_agc_pm[b][t] + self.generator_sampling_time / self.generator_time_constants[self.map_bus_generators[b]] * ((gp_agc_pr[b][t + 1] - (gp_agc_f[b][t + 1] - gp_agc_f[b][0]) ) * 20 - gp_agc_pm[b][t + 1]) )

        # [AGC] nominal frequency constraint
        for b in range(1, self.bus_system + 1):
            model.addConstr(gp_agc_f[b][0] == 1)
        
        # [AGC] frequency constraints for generators with governors
        for t in range(self.number_timeslots - 1):
            for b in self.generator_buses:
                model.addConstr(gp_agc_f[b][t + 1] == gp_agc_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_agc_pm[b][t + 1] - 0.1 * gp_agc_pg[b][t + 1]))
        
        
        # [AGC] rotor angle initialization for slack buses (reference bus initialization)
        for b in self.slack_buses:
            model.addConstr(gp_agc_delta[b][0] == self.delta_ref[b] * (math.pi / 180))
        
        # [AGC] phase angle constraints
        for t in range(self.number_timeslots - 1):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_agc_delta[b][t + 1] == gp_agc_delta[b][t] +  2 * math.pi * self.generator_sampling_time * (gp_agc_f[b][t + 1] - gp_agc_f[b][0]))
        
        # [AGC] reference setpoint for initial timeslot
        for b in self.generator_with_governors:
            model.addConstr(gp_agc_pr[b][0] == gp_agc_pg[b][0] * 0.005)

        # [AGC] reference setpoint for different generators        
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                if t % self.agc_cycle == 0:
                    model.addConstr(gp_agc_pr[b][t] == gp_agc_pg[b][t] * 0.005)
                else:
                    model.addConstr(gp_agc_pr[b][t] == gp_agc_pr[b][t - 1])
    
        # [AGC] calculating system frequency                    
        sum_freq = 0
        for b in self.generator_buses:
            sum_freq += gp_gen_f[b][self.number_timeslots - 2]
            
        model.addConstr(gp_sys_f == sum_freq / len(self.generator_buses))
        

        model.optimize()
        
        
        self.system_frequency = gp_sys_f.x * 60
        self.execution_time = time.time() - start_time
        
        print('System Frequency:', self.system_frequency)        
        print('Model Status:', model.Status)
        print('Execution Time', time.time() - start_time)

        dictionary = dict()
        dictionary['PL']    = [[gp_load[i][j].x for j in range(len(gp_load[0]))] for i in range(len(gp_load))] 
        dictionary['PG']    = [[gp_gen_pg[i][j].x for j in range(len(gp_gen_pg[0]))] for i in range(len(gp_gen_pg))]
        dictionary['PM']    = [[gp_gen_pm[i][j].x for j in range(len(gp_gen_pm[0]))] for i in range(len(gp_gen_pm))]
        dictionary['Delta'] = [[gp_gen_delta[i][j].x for j in range(len(gp_gen_delta[0]))] for i in range(len(gp_gen_delta))]
        dictionary['F']     = [[gp_gen_f[i][j].x for j in range(len(gp_gen_f[0]))] for i in range(len(gp_gen_f))]
        
        dictionary['AGC-PG']    = [[gp_agc_pg[i][j].x for j in range(len(gp_agc_pg[0]))] for i in range(len(gp_agc_pg))]
        dictionary['AGC-PM']    = [[gp_agc_pm[i][j].x for j in range(len(gp_agc_pm[0]))] for i in range(len(gp_agc_pm))]
        dictionary['AGC-PR']    = [[gp_agc_pr[i][j].x for j in range(len(gp_agc_pr[0]))] for i in range(len(gp_agc_pr))]
        dictionary['AGC-Delta'] = [[gp_agc_delta[i][j].x for j in range(len(gp_agc_delta[0]))] for i in range(len(gp_agc_delta))]
        dictionary['AGC-F']     = [[gp_agc_f[i][j].x for j in range(len(gp_agc_f[0]))] for i in range(len(gp_agc_f))]
        
        dictionary['Sys-F'] = self.system_frequency
        dictionary['C']     = model.NumConstrs
        dictionary['T']     = self.execution_time
        
        with open('output-files/' + filename_to_save_output, "w") as outfile:
            json.dump(dictionary, outfile)    
        
        gp_load        = [[gp_load[i][j].x for j in range(len(gp_load[0]))] for i in range(len(gp_load))] 
        gp_gen_pg      = [[gp_gen_pg[i][j].x for j in range(len(gp_gen_pg[0]))] for i in range(len(gp_gen_pg))]
        gp_gen_pm      = [[gp_gen_pm[i][j].x for j in range(len(gp_gen_pm[0]))] for i in range(len(gp_gen_pm))]
        gp_gen_delta   = [[gp_gen_delta[i][j].x for j in range(len(gp_gen_delta[0]))] for i in range(len(gp_gen_delta))]
        gp_gen_f       = [[gp_gen_f[i][j].x for j in range(len(gp_gen_f[0]))] for i in range(len(gp_gen_f))]
        
        gp_agc_f   = [[gp_agc_f[i][j].x for j in range(len(gp_agc_f[0]))] for i in range(len(gp_agc_f))]
        gp_agc_pg  = [[gp_agc_pg[i][j].x for j in range(len(gp_agc_pg[0]))] for i in range(len(gp_agc_pg))]
        gp_agc_pr  = [[gp_agc_pr[i][j].x for j in range(len(gp_agc_pr[0]))] for i in range(len(gp_agc_pr))]
        
        self.visualize("", gp_load, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{L}~\mathit{(p.u.)}$ ", self.attackable_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{L}_{' + str(b) + r'}$') for b in self.attackable_buses], 'upper right', '', 'plots/Case-Study-2_Load')
        self.visualize("", gp_gen_pg, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{G}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{G}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Generating-Power')
        self.visualize("", gp_gen_pm, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{M}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{M}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Mechanical-Power')
        self.visualize("", gp_agc_pr, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{R}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{R}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Reference-Power')
        self.visualize("", gp_gen_delta, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\delta~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\delta_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Rotor-Angle')
        self.visualize("", gp_gen_f, 60, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{F}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')
        self.visualize("", gp_agc_f, 60, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{F}^\mathit{AGC}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')              

    def attack_vector_optimization(self, number_timeslots, timeslot_benign_load, max_buses_to_attack, attack_goal, type_defense, filename_to_save_output):
        
        start_time = time.time()
        
        self.number_timeslots        = number_timeslots
        self.number_agc_steps        = int(self.number_timeslots / self.agc_cycle)
        
        model = gp.Model()
        
        # Initializing AGC measurement variables
        gp_agc_pg    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='gp_agc_pg_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pl    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pl_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pr    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pr_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pm    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pm_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_delta = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_delta_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_f     = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_f_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]

        # Initializing system measurement variables
        gp_gen_pg    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='gp_gen_pg_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_pl    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_pl_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_pm    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_pm_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_delta = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_delta_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_f     = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_f_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_sys_f  = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'gp_sys_f')

        sum_gp_agc_attacked_load = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'sum_gp_agc_attacked_load')
        
        ### Attack parameters
        av_load  = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='av_load_' + str(t) + '_' + str(b)) for t in range(self.number_agc_steps)] for b in range(self.bus_system + 1)]
        bus_attackable = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.BINARY, name='bus_attackable_' + str(b)) for b in range(self.bus_system + 1)]
        
        gp_load          = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_load_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_attacked_load = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_attacked_load_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_attacked_load = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_attacked_load_' + str(t) + '_' + str(b)) for t in range(self.number_agc_steps)] for b in range(self.bus_system + 1)]
        
        inference_variables = []
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:
                inference_variables.append([[[model.addVar(lb = 0, ub = 1, vtype = gp.GRB.BINARY, name='inference_variables_' + str(t) + '_' + str(c) + '_' + str(e) + '_' + str(b)) for e in range(len(self.all_hulls[b][c]))] for c in range(len(self.all_hulls[b]))] for t in range(self.number_agc_steps)])
            else:
                inference_variables.append(-1)
            
        indication_variables = []
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:
                indication_variables.append([[model.addVar(lb = 0, ub = 1, vtype = gp.GRB.BINARY, name='indication_variables_' + str(t) + '_' + str(c) + '_' + str(b)) for c in range(len(self.all_hulls[b]))] for t in range(self.number_agc_steps + 1)])
            else:
                indication_variables.append(-1)
            
        decision_variables = []
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:
                decision_variables.append([model.addVar(lb = 0, ub = 1, vtype = gp.GRB.BINARY, name='decision_variables_' + str(t) + '_' + str(b)) for t in range(self.number_agc_steps + 1)] )
            else:
                decision_variables.append(-1)
        
        
        max_freq = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'max_freq')
        max_freqs     = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='max_freqs_' + str(t)) for t in range(self.number_agc_steps)]
        
        min_freq = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'min_freq')
        min_freqs     = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='min_freqs_' + str(t)) for t in range(self.number_agc_steps)]
        

        num_attack_cycle = model.addVar(lb = 0, ub = self.number_agc_steps, vtype = gp.GRB.INTEGER, name = 'num_attack_cycle')

        inference_gt = [model.addVar(vtype = gp.GRB.BINARY, name='inference_gt_' + str(t)) for t in range(self.number_agc_steps)]
        inference_lt = [model.addVar(vtype = gp.GRB.BINARY, name='inference_lt_' + str(t)) for t in range(self.number_agc_steps)]
        inference_attack_cycle = [model.addVar(vtype = gp.GRB.BINARY, name='inference_attack_cycle_' + str(t)) for t in range(self.number_agc_steps)]
           
        agc_attackable = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.BINARY, name='agc_attackable_' + str(t)) for t in range(self.number_agc_steps)]

        or_variables = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.BINARY, name='or_variables_' + str(t) + '_' + str(b)) for b in range(self.bus_system + 1)] for t in range(self.number_agc_steps)]
        or_constraints = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='or_constraints_' + str(t)) for t in range(self.number_agc_steps)]

        
        #######################################################################
        ########################## System dynamics ############################
        #######################################################################      
        
        # [BUS] Benign load measurements
        for t in range(self.number_timeslots):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_load[b][t] == self.benign_loads[b][timeslot_benign_load])            
        
        # [BUS] Actual power flow 
        for t in range(self.number_timeslots):
            for b_1 in range(1, self.bus_system + 1):
                to_add = 0
                for b_2 in range(1, self.bus_system + 1):
                    to_add += self.bus_susceptance_matrix[b_1 - 1][b_2 - 1] * (gp_gen_delta[b_1][t] - gp_gen_delta[b_2][t])  
                
                model.addConstr(gp_gen_pg[b_1][t] - gp_load[b_1][t] == to_add)
        
        
        # [GEN] active power output for pg buses in the initial timeslot
        for b in self.pv_buses:
            model.addConstr(gp_gen_pg[b][0] == self.pg_ref[b])
            
                
        # [GEN] active power output for for pq buses      
        for b in self.pq_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pg[b][t] == 0)
                
        # [GEN] swing equation constraints for generators with no governors
        for b in self.generator_with_no_governors:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pm[b][t] == gp_gen_pg[b][0] * 0.1)
        
        # [GEN] initial timeslot swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            model.addConstr(gp_gen_pm[b][0] == gp_agc_pr[b][0] * 20)
        
        # [GEN] other timeslots swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                model.addConstr(gp_gen_pm[b][t + 1] == gp_gen_pm[b][t] + self.generator_sampling_time / self.generator_time_constants[self.map_bus_generators[b]] * ((gp_agc_pr[b][t + 1] - (gp_gen_f[b][t + 1] - gp_gen_f[b][0])) * 20 - gp_gen_pm[b][t + 1]) )
        
        # [BUS] nominal frequency constraint
        for b in range(1, self.bus_system + 1):
            model.addConstr(gp_gen_f[b][0] == 1)
        
        # [BUS] frequency constraints for generators with governors
        for t in range(self.number_timeslots - 1):
            for b in self.generator_buses:
                model.addConstr(gp_gen_f[b][t + 1] == gp_gen_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_gen_pm[b][t + 1] - 0.1 * gp_gen_pg[b][t + 1]))
                
        # [GEN] rotor angle initialization for slack buses (reference bus initialization)
        for b in self.slack_buses:
            model.addConstr(gp_gen_delta[b][0] == self.delta_ref[b] * (math.pi / 180))
                
        # [BUS] phase angle constraints
        for t in range(self.number_timeslots - 1):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_gen_delta[b][t + 1] == gp_gen_delta[b][t] + 2 * math.pi * self.generator_sampling_time * (gp_gen_f[b][t + 1] - gp_gen_f[b][0]))

        #######################################################################
        ########### Automatic Generation Control Controller ###################
        #######################################################################
        
        # [AGC] AGC Cycle attacked load measurements
        for t in range(self.number_agc_steps):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_agc_attacked_load[b][t] == gp_load[b][t * self.agc_cycle] + av_load[b][t])

        # [AGC] All timeslot attacked load measurements
        for t in range(self.number_timeslots):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_attacked_load[b][t] == gp_load[b][t] + av_load[b][int(t / self.agc_cycle)])


        # [AGC] Power flow analysis
        for t in range(self.number_timeslots):
            for b_1 in range(1, self.bus_system + 1):
                to_add = 0

                for b_2 in range(1, self.bus_system + 1):
                    to_add += self.bus_susceptance_matrix[b_1 - 1][b_2 - 1] * (gp_agc_delta[b_1][t] - gp_agc_delta[b_2][t])
                
                model.addConstr(gp_agc_pg[b_1][t] - gp_agc_attacked_load[b_1][int(t / self.agc_cycle)] == to_add)

        # AGC ADM-rules for initial cycle
        for t in range(self.number_agc_steps):
            for b in range(1, self.bus_system + 1):
                if t == 0 or b not in self.attackable_buses:
                    model.addConstr(av_load[b][t] == 0)
        
        
        # [AGC] ADM rules
        eps = 0.0000000001
        M = 10000 + eps
        
        if type_defense == 'ADM':
            for b in self.attackable_buses:
                for t in range(1, self.number_agc_steps):
                    for c in range(len(self.all_hulls[b])):
                        for e in range(len(self.all_hulls[b][c])):
                            current_val = gp_agc_attacked_load[b][t - 1] * self.all_hulls[b][c][e][0] + (gp_agc_attacked_load[b][t] - gp_agc_attacked_load[b][t - 1]) * self.all_hulls[b][c][e][1] + self.all_hulls[b][c][e][2]
                            model.addConstr(current_val >= eps - M * (inference_variables[b][t][c][e]))
                            model.addConstr(current_val <= M * (1 - inference_variables[b][t][c][e]))    
        
                        model.addGenConstrAnd(indication_variables[b][t][c], inference_variables[b][t][c])
                    model.addGenConstrOr(decision_variables[b][t], indication_variables[b][t])
                    model.addConstr(decision_variables[b][t] == 1)
 
        elif type_defense == 'BDD':
            # [AGC] BDD rules    
            for t in range(self.number_agc_steps):
                for b in range(1, self.bus_system + 1):
                    model.addConstr(gp_agc_attacked_load[b][t] - gp_agc_attacked_load[b][t - 1] <= self.load_thresholds_pos[b])
                    model.addConstr(gp_agc_attacked_load[b][t] - gp_agc_attacked_load[b][t - 1]  >= self.load_thresholds_neg[b])
                    
                    model.addConstr(gp_agc_attacked_load[b][t] >= 0)
        
        # [AGC] active power output for pv bus generators
        for b in self.pv_buses:
            model.addConstr(gp_agc_pg[b][0] == self.pg_ref[b])    
    
        # [AGC] active power output for for pq buses      
        for b in self.pq_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_agc_pg[b][t] == 0)  
                
        # [AGC] swing equation constraints for generators with no governors
        for b in self.generator_with_no_governors:
            for t in range(self.number_timeslots):
                model.addConstr(gp_agc_pm[b][t] == gp_agc_pg[b][0] * 0.1)
        
        # [AGC] initial timeslot swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            model.addConstr(gp_agc_pm[b][0] == gp_agc_pr[b][0] * 20)
        
        # [AGC] other timeslots swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                model.addConstr(gp_agc_pm[b][t + 1] == gp_agc_pm[b][t] + self.generator_sampling_time / self.generator_time_constants[self.map_bus_generators[b]] * ((gp_agc_pr[b][t + 1] - (gp_agc_f[b][t + 1] - gp_agc_f[b][0]) ) * 20 - gp_agc_pm[b][t + 1]) )

        # [AGC] nominal frequency constraint
        for b in range(1, self.bus_system + 1):
            model.addConstr(gp_agc_f[b][0] == 1)
        
        # [AGC] frequency constraints for generators with governors
        for t in range(self.number_timeslots - 1):
            for b in self.generator_buses:
                model.addConstr(gp_agc_f[b][t + 1] == gp_agc_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_agc_pm[b][t + 1] - 0.1 * gp_agc_pg[b][t + 1]))
                #model.addConstr( (bus_attackable[b] == 1) >> (gp_agc_f[b][t + 1] == gp_agc_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_agc_pm[b][t + 1] - 0.1 * gp_agc_pg[b][t + 1])))
                #model.addConstr( (bus_attackable[b] == 0) >> (gp_agc_f[b][t + 1] == gp_gen_f[b][t + 1]))
        
        # [AGC] rotor angle initialization for slack buses (reference bus initialization)
        for b in self.slack_buses:
            model.addConstr(gp_agc_delta[b][0] == self.delta_ref[b] * (math.pi / 180))
        
        # [AGC] phase angle constraints
        for t in range(self.number_timeslots - 1):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_agc_delta[b][t + 1] == gp_agc_delta[b][t] +  2 * math.pi * self.generator_sampling_time * (gp_agc_f[b][t + 1] - gp_agc_f[b][0]))
        
        # [AGC] reference setpoint for initial timeslot
        for b in self.generator_with_governors:
            model.addConstr(gp_agc_pr[b][0] == gp_agc_pg[b][0] * 0.005)

        # [AGC] reference setpoint for different generators        
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                if t % self.agc_cycle == 0:
                    model.addConstr(gp_agc_pr[b][t] == gp_agc_pg[b][t] * 0.005)
                else:
                    model.addConstr(gp_agc_pr[b][t] == gp_agc_pr[b][t - 1])
        
        # [AGC] attacker's capability constraint of attackable buses
        for t in range(self.number_agc_steps):
            for b in range(1, self.bus_system + 1):
                model.addConstr( (bus_attackable[b] == 0) >> (av_load[b][t] == 0) )
                                
        for b in range(self.bus_system + 1):
            if b not in self.attackable_buses:
                model.addConstr(bus_attackable[b] == 0)
        
        
        model.addConstr(sum(bus_attackable) == max_buses_to_attack)
        
    
        #######################################################################
        ####################### attack constraints ###########################
        ######################################################################                         
        big_M = 10000
        for i in range(self.number_agc_steps):
            
            model.addConstr( num_attack_cycle >= (i + 1) - big_M * inference_gt[i] )
            model.addConstr( num_attack_cycle <= (i + 1) + big_M * ( 1- inference_gt[i]) )
            
            model.addConstr( num_attack_cycle >= (i - 1) - big_M * ( 1- inference_lt[i]) )
            model.addConstr( num_attack_cycle <= (i - 1) + big_M * inference_lt[i] )
            
            model.addGenConstrAnd(inference_attack_cycle[i], [inference_gt[i], inference_lt[i]])
        
        
        for i in range(self.number_agc_steps):
            for j in range(self.number_agc_steps):
                if i >= j:
                    model.addConstr( (inference_attack_cycle[i] == 1) >>  (agc_attackable[j] == 1) )
                else:
                    model.addConstr( (inference_attack_cycle[i] == 1) >>  (agc_attackable[j] == 0) )
            
        #model.addConstr(num_attack_cycle >= 1)
        #model.addConstr(num_attack_cycle <= self.number_agc_steps - 1)
        model.addConstr(num_attack_cycle == self.number_agc_steps - 1)
          
        
        if attack_goal == 'UF':
            
            for t in range(self.number_agc_steps):
                or_ = []
                for b in self.generator_buses:    
                    model.addConstr(min_freqs[t] <= gp_gen_f[b][(t + 1) * self.agc_cycle - 2])
                    model.addConstr( (or_variables[t][b] == 1) >> (min_freqs[t] == gp_gen_f[b][(t + 1) * self.agc_cycle - 2]) )
                
                for b in range(self.bus_system + 1):
                    if b not in self.generator_buses:
                        model.addConstr(or_variables[t][b] == 0)
              
                model.addGenConstrOr(or_constraints[t] , or_variables[t])
                model.addConstr(or_constraints[t] == 1)
                
                model.addConstr( (inference_attack_cycle[t] == 1) >>  (min_freq == min_freqs[t] * 60) ) 

            model.addConstr(min_freq <= self.uf_threshold)
        
        elif attack_goal == 'OF':
            
            for t in range(self.number_agc_steps):
                or_ = []
                for b in self.generator_buses:    
                    model.addConstr(max_freqs[t] >= gp_gen_f[b][(t + 1) * self.agc_cycle - 2])
                    model.addConstr( (or_variables[t][b] == 1) >> (max_freqs[t] == gp_gen_f[b][(t + 1) * self.agc_cycle - 2]) )
                
                for b in range(self.bus_system + 1):
                    if b not in self.generator_buses:
                        model.addConstr(or_variables[t][b] == 0)
                
                model.addGenConstrOr(or_constraints[t] , or_variables[t])
                model.addConstr(or_constraints[t] == 1)
                
                model.addConstr( (inference_attack_cycle[t] == 1) >>  (max_freq == max_freqs[t] * 60) )            
            #model.addConstr(max_freq >= self.of_threshold)
            
        #model.setObjective(num_attack_cycle, gp.GRB.MINIMIZE)       
        model.setObjective(max_freq, gp.GRB.MAXIMIZE)       


        model.optimize()
        
        self.execution_time = time.time() - start_time
        
        print("Model Status:", model.Status)
        print("MAX_FREQ", max_freq.x)
        if attack_goal == 'UF':
            print("Minimum Frequency:", min_freq.x)
        elif attack_goal == 'OF':
            print("Maximum Frequency:", max_freq.x)
        print("Number of attack cycle", num_attack_cycle.x)
                    
        self.attack_time  = num_attack_cycle.x
        self.number_const = model.NumConstrs
        
        # Finding attack bus number
        attacked_bus = -1
        for b in self.generator_buses: 
            if int(or_variables[int(self.attack_time)][b].x) == 1:
                attacked_bus = b
                break
        
        print("Attacked bus", attacked_bus)
        print("Execution time", time.time() - start_time)
        print("Number of Constraints", model.NumConstrs)
        
        dictionary = dict()
        dictionary['PL']    = [[gp_load[i][j].x for j in range(len(gp_load[0]))] for i in range(len(gp_load))] 
        dictionary['PL-A']   = [[gp_attacked_load[i][j].x for j in range(len(gp_attacked_load[0]))] for i in range(len(gp_attacked_load))] 
        dictionary['PG']    = [[gp_gen_pg[i][j].x for j in range(len(gp_gen_pg[0]))] for i in range(len(gp_gen_pg))]
        dictionary['PM']    = [[gp_gen_pm[i][j].x for j in range(len(gp_gen_pm[0]))] for i in range(len(gp_gen_pm))]
        dictionary['Delta'] = [[gp_gen_delta[i][j].x for j in range(len(gp_gen_delta[0]))] for i in range(len(gp_gen_delta))]
        dictionary['F']     = [[gp_gen_f[i][j].x for j in range(len(gp_gen_f[0]))] for i in range(len(gp_gen_f))]
        
        dictionary['AGC-PG']    = [[gp_agc_pg[i][j].x for j in range(len(gp_agc_pg[0]))] for i in range(len(gp_agc_pg))]
        dictionary['AGC-PM']    = [[gp_agc_pm[i][j].x for j in range(len(gp_agc_pm[0]))] for i in range(len(gp_agc_pm))]
        dictionary['AGC-PR']    = [[gp_agc_pr[i][j].x for j in range(len(gp_agc_pr[0]))] for i in range(len(gp_agc_pr))]
        dictionary['AGC-Delta'] = [[gp_agc_delta[i][j].x for j in range(len(gp_agc_delta[0]))] for i in range(len(gp_agc_delta))]
        dictionary['AGC-F']     = [[gp_agc_f[i][j].x for j in range(len(gp_agc_f[0]))] for i in range(len(gp_agc_f))]
        
        dictionary['Max-F'] = max_freq.x
        dictionary['C']     = model.NumConstrs
        dictionary['AT']    = self.attack_time
        dictionary['AB']    = int(attacked_bus)
        dictionary['T']     = self.execution_time
        
        with open('output-files/' + filename_to_save_output, "w") as outfile:
            json.dump(dictionary, outfile)    
            
        gp_load        = [[gp_load[i][j].x for j in range(len(gp_load[0]))] for i in range(len(gp_load))] 
        gp_gen_pg      = [[gp_gen_pg[i][j].x for j in range(len(gp_gen_pg[0]))] for i in range(len(gp_gen_pg))]
        gp_gen_pm      = [[gp_gen_pm[i][j].x for j in range(len(gp_gen_pm[0]))] for i in range(len(gp_gen_pm))]
        gp_gen_delta   = [[gp_gen_delta[i][j].x for j in range(len(gp_gen_delta[0]))] for i in range(len(gp_gen_delta))]
        gp_gen_f       = [[gp_gen_f[i][j].x for j in range(len(gp_gen_f[0]))] for i in range(len(gp_gen_f))]
        
        gp_agc_f              = [[gp_agc_f[i][j].x for j in range(len(gp_agc_f[0]))] for i in range(len(gp_agc_f))]
        gp_agc_pg             = [[gp_agc_pg[i][j].x for j in range(len(gp_agc_pg[0]))] for i in range(len(gp_agc_pg))]
        gp_agc_pr             = [[gp_agc_pr[i][j].x for j in range(len(gp_agc_pr[0]))] for i in range(len(gp_agc_pr))]
        gp_agc_load    =  [[gp_attacked_load[i][j].x for j in range(len(gp_attacked_load[0]))] for i in range(len(gp_attacked_load))] 
             
        
        self.visualize("", gp_load, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{L}~\mathit{(p.u.)}$ ", self.attackable_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{L}_{' + str(b) + r'}$') for b in self.attackable_buses], 'upper right', '', 'plots/Case-Study-2_Load')
        self.visualize("", gp_gen_pg, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{G}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{G}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Generating-Power')
        self.visualize("", gp_gen_pm, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{M}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{M}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Mechanical-Power')
        self.visualize("", gp_agc_pr, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{R}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{R}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Reference-Power')
        self.visualize("", gp_gen_delta, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\delta~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\delta_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Rotor-Angle')
        self.visualize("", gp_gen_f, 60, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{F}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')
        self.visualize("", gp_agc_load, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\bar{\mathcal{P}}^L~\mathit{(p.u.)}$ ", self.attackable_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.attackable_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')
        self.visualize("", gp_agc_f, 60, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{F}^\mathit{AGC}~\mathit{(p.u.)}$ ", self.pv_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.pv_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')              
        
        if type_defense == 'ADM':
            for b in self.attackable_buses:
                plt.figure()
                plt.xlabel('Load Measurement (P.U.) at Time, t - 1')
                plt.ylabel('Deviation Measurement (P.U.) at Time, t')
                plt.title('ADM for Bus' + str(b))
                for c in range(len(self.all_vertices[b])):
                    plt.plot(self.all_vertices[b][c][:, 0], self.all_vertices[b][c][:, 1])
                   
                    #for e in range(len(self.all_vertices[b][c])):
                    #    #print(b, c, e, self.all_vertices[b][c][e])
                    #    plt.scatter(self.all_vertices[b][c][:, 0], self.all_vertices[b][c][:, 1], s = 10)
            
                for t in range(1, self.number_agc_steps):
                    plt.scatter(gp_agc_attacked_load[b][(t - 1)].x, gp_agc_attacked_load[b][t].x - gp_agc_attacked_load[b][t - 1].x,  s = 50)
                
    def resiliency_analysis(self, number_timeslots, timeslot_benign_load, attack_goal, type_defense, filename_to_save_output):
        
        start_time = time.time()
        
        self.number_timeslots        = number_timeslots
        self.number_agc_steps        = int(self.number_timeslots / self.agc_cycle)
        
        model = gp.Model()
        
        # Initializing AGC measurement variables
        gp_agc_pg    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='gp_agc_pg_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pl    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pl_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pr    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pr_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_pm    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_pm_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_delta = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_delta_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_f     = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_agc_f_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]

        # Initializing system measurement variables
        gp_gen_pg    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='gp_gen_pg_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_pl    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_pl_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_pm    = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_pm_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_delta = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_delta_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_gen_f     = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_gen_f_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_sys_f  = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'gp_sys_f')

        sum_gp_agc_attacked_load = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'sum_gp_agc_attacked_load')
        
        
        ### Attack parameters
        av_load  = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='av_load_' + str(t) + '_' + str(b)) for t in range(self.number_agc_steps)] for b in range(self.bus_system + 1)]
        bus_attackable = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.BINARY, name='bus_attackable_' + str(b)) for b in range(self.bus_system + 1)]
        
        gp_load          = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_load_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_attacked_load = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_attacked_load_' + str(t) + '_' + str(b)) for t in range(self.number_timeslots)] for b in range(self.bus_system + 1)]
        gp_agc_attacked_load = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS, name='gp_attacked_load_' + str(t) + '_' + str(b)) for t in range(self.number_agc_steps)] for b in range(self.bus_system + 1)]
        
        inference_variables = []
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:
                inference_variables.append([[[model.addVar(lb = 0, ub = 1, vtype = gp.GRB.BINARY, name='inference_variables_' + str(t) + '_' + str(c) + '_' + str(e) + '_' + str(b)) for e in range(len(self.all_hulls[b][c]))] for c in range(len(self.all_hulls[b]))] for t in range(self.number_agc_steps)])
            else:
                inference_variables.append(-1)
            
        indication_variables = []
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:
                indication_variables.append([[model.addVar(lb = 0, ub = 1, vtype = gp.GRB.BINARY, name='indication_variables_' + str(t) + '_' + str(c) + '_' + str(b)) for c in range(len(self.all_hulls[b]))] for t in range(self.number_agc_steps + 1)])
            else:
                indication_variables.append(-1)
            
        decision_variables = []
        for b in range(self.bus_system + 1):
            if b in self.attackable_buses:
                decision_variables.append([model.addVar(lb = 0, ub = 1, vtype = gp.GRB.BINARY, name='decision_variables_' + str(t) + '_' + str(b)) for t in range(self.number_agc_steps + 1)] )
            else:
                decision_variables.append(-1)
        
        
        max_freq = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'max_freq')
        max_freqs     = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='max_freqs_' + str(t)) for t in range(self.number_agc_steps)]
        
        min_freq = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'min_freq')
        min_freqs     = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='min_freqs_' + str(t)) for t in range(self.number_agc_steps)]
        
        num_attack_cycle = model.addVar(lb = 0, ub = self.number_agc_steps, vtype = gp.GRB.INTEGER, name = 'num_attack_cycle')


        inference_gt = [model.addVar(vtype = gp.GRB.BINARY, name='inference_gt_' + str(t)) for t in range(self.number_agc_steps)]
        inference_lt = [model.addVar(vtype = gp.GRB.BINARY, name='inference_lt_' + str(t)) for t in range(self.number_agc_steps)]
        inference_attack_cycle = [model.addVar(vtype = gp.GRB.BINARY, name='inference_attack_cycle_' + str(t)) for t in range(self.number_agc_steps)]
           
        agc_attackable = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.BINARY, name='agc_attackable_' + str(t)) for t in range(self.number_agc_steps)]

        or_variables = [[model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY,  vtype=gp.GRB.BINARY, name='or_variables_' + str(t) + '_' + str(b)) for b in range(self.bus_system + 1)] for t in range(self.number_agc_steps)]
        or_constraints = [model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='or_constraints_' + str(t)) for t in range(self.number_agc_steps)]

        num_attacked_buses = model.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = 'num_atacked_buses')

        #######################################################################
        ########################## System dynamics ############################
        #######################################################################      
        
        # [BUS] Benign load measurements
        for t in range(self.number_timeslots):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_load[b][t] == self.benign_loads[b][timeslot_benign_load])            
        
        # [BUS] Actual power flow 
        for t in range(self.number_timeslots):
            for b_1 in range(1, self.bus_system + 1):
                to_add = 0
                for b_2 in range(1, self.bus_system + 1):
                    to_add += self.bus_susceptance_matrix[b_1 - 1][b_2 - 1] * (gp_gen_delta[b_1][t] - gp_gen_delta[b_2][t])  
                
                model.addConstr(gp_gen_pg[b_1][t] - gp_load[b_1][t] == to_add)
        
        # [GEN] active power output for pg buses in the initial timeslot
        for b in self.pv_buses:
            model.addConstr(gp_gen_pg[b][0] == self.pg_ref[b])
                
        # [GEN] active power output for for pq buses      
        for b in self.pq_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pg[b][t] == 0)
                
        # [GEN] swing equation constraints for generators with no governors
        for b in self.generator_with_no_governors:
            for t in range(self.number_timeslots):
                model.addConstr(gp_gen_pm[b][t] == gp_gen_pg[b][0] * 0.1)
        
        # [GEN] initial timeslot swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            model.addConstr(gp_gen_pm[b][0] == gp_agc_pr[b][0] * 20)
        
        # [GEN] other timeslots swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                model.addConstr(gp_gen_pm[b][t + 1] == gp_gen_pm[b][t] + self.generator_sampling_time / self.generator_time_constants[self.map_bus_generators[b]] * ((gp_agc_pr[b][t + 1] - (gp_gen_f[b][t + 1] - gp_gen_f[b][0])) * 20 - gp_gen_pm[b][t + 1]) )
        
        # [BUS] nominal frequency constraint
        for b in range(1, self.bus_system + 1):
            model.addConstr(gp_gen_f[b][0] == 1)
        
        # [BUS] frequency constraints for generators with governors
        for t in range(self.number_timeslots - 1):
            for b in self.generator_buses:
                model.addConstr(gp_gen_f[b][t + 1] == gp_gen_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_gen_pm[b][t + 1] - 0.1 * gp_gen_pg[b][t + 1]))
        
        # [GEN] rotor angle initialization for slack buses (reference bus initialization)
        for b in self.slack_buses:
            model.addConstr(gp_gen_delta[b][0] == self.delta_ref[b] * (math.pi / 180))
                
        # [BUS] phase angle constraints
        for t in range(self.number_timeslots - 1):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_gen_delta[b][t + 1] == gp_gen_delta[b][t] + 2 * math.pi * self.generator_sampling_time * (gp_gen_f[b][t + 1] - gp_gen_f[b][0]))

        #######################################################################
        ########### Automatic Generation Control Controller ###################
        #######################################################################
        
        # [AGC] AGC Cycle attacked load measurements
        for t in range(self.number_agc_steps):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_agc_attacked_load[b][t] == gp_load[b][t * self.agc_cycle] + av_load[b][t])

        # [AGC] All timeslot attacked load measurements
        for t in range(self.number_timeslots):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_attacked_load[b][t] == gp_load[b][t] + av_load[b][int(t / self.agc_cycle)])
        

        # [AGC] Power flow analysis
        for t in range(self.number_timeslots):
            for b_1 in range(1, self.bus_system + 1):
                to_add = 0
                for b_2 in range(1, self.bus_system + 1):
                    to_add += self.bus_susceptance_matrix[b_1 - 1][b_2 - 1] * (gp_agc_delta[b_1][t] - gp_agc_delta[b_2][t])  
                
                model.addConstr(gp_agc_pg[b_1][t] - gp_agc_attacked_load[b_1][int(t / self.agc_cycle)] == to_add)
        
        # AGC ADM-rules for initial cycle
        for t in range(self.number_agc_steps):
            for b in range(1, self.bus_system + 1):
                if t == 0 or b not in self.attackable_buses:
                    model.addConstr(av_load[b][t] == 0)
        
        # [AGC] ADM rules
        eps = 0.0000000001
        M = 10000 + eps
        
        if type_defense == 'ADM':
            for b in self.attackable_buses:
                for t in range(1, self.number_agc_steps):
                    for c in range(len(self.all_hulls[b])):
                        for e in range(len(self.all_hulls[b][c])):
                            current_val = gp_agc_attacked_load[b][t - 1] * self.all_hulls[b][c][e][0] + (gp_agc_attacked_load[b][t] - gp_agc_attacked_load[b][t - 1]) * self.all_hulls[b][c][e][1] + self.all_hulls[b][c][e][2]
                            model.addConstr(current_val >= eps - M * (inference_variables[b][t][c][e]))
                            model.addConstr(current_val <= M * (1 - inference_variables[b][t][c][e]))    
        
                        model.addGenConstrAnd(indication_variables[b][t][c], inference_variables[b][t][c])
                    model.addGenConstrOr(decision_variables[b][t], indication_variables[b][t])
                    model.addConstr(decision_variables[b][t] == 1)

        elif type_defense == 'BDD':
            # [AGC] BDD rules    
            for t in range(self.number_agc_steps):
                for b in range(1, self.bus_system + 1):
                    model.addConstr(gp_agc_attacked_load[b][t] - gp_agc_attacked_load[b][t - 1] <= self.load_thresholds_pos[b])
                    model.addConstr(gp_agc_attacked_load[b][t] - gp_agc_attacked_load[b][t - 1]  >= self.load_thresholds_neg[b])
                    
                    model.addConstr(gp_agc_attacked_load[b][t] >= 0)
        
        # [AGC] active power output for pv bus generators
        for b in self.pv_buses:
            model.addConstr(gp_agc_pg[b][0] == self.pg_ref[b])    
    
        # [AGC] active power output for for pq buses      
        for b in self.pq_buses:
            for t in range(self.number_timeslots):
                model.addConstr(gp_agc_pg[b][t] == 0)  
                
        # [AGC] swing equation constraints for generators with no governors
        for b in self.generator_with_no_governors:
            for t in range(self.number_timeslots):
                model.addConstr(gp_agc_pm[b][t] == gp_agc_pg[b][0] * 0.1)
        
        # [AGC] initial timeslot swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            model.addConstr(gp_agc_pm[b][0] == gp_agc_pr[b][0] * 20)
        
        # [AGC] other timeslots swing equation constraints for generators with governors
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                model.addConstr(gp_agc_pm[b][t + 1] == gp_agc_pm[b][t] + self.generator_sampling_time / self.generator_time_constants[self.map_bus_generators[b]] * ((gp_agc_pr[b][t + 1] - (gp_agc_f[b][t + 1] - gp_agc_f[b][0]) ) * 20 - gp_agc_pm[b][t + 1]) )

        # [AGC] nominal frequency constraint
        for b in range(1, self.bus_system + 1):
            model.addConstr(gp_agc_f[b][0] == 1)
        
        # [AGC] frequency constraints for generators with governors
        for t in range(self.number_timeslots - 1):
            for b in self.generator_buses:
                
                
                model.addConstr(gp_agc_f[b][t + 1] == gp_agc_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_agc_pm[b][t + 1] - 0.1 * gp_agc_pg[b][t + 1]))
                #model.addConstr( (bus_attackable[b] == 1) >> (gp_agc_f[b][t + 1] == gp_agc_f[b][t] + (self.generator_sampling_time / (2 * self.generator_inertia[self.map_bus_generators[b]])) * (gp_agc_pm[b][t + 1] - 0.1 * gp_agc_pg[b][t + 1])))
                #model.addConstr( (bus_attackable[b] == 0) >> (gp_agc_f[b][t + 1] == gp_gen_f[b][t + 1]))
        
                
        # [AGC] rotor angle initialization for slack buses (reference bus initialization)
        for b in self.slack_buses:
            model.addConstr(gp_agc_delta[b][0] == self.delta_ref[b] * (math.pi / 180))
        
        # [AGC] phase angle constraints
        for t in range(self.number_timeslots - 1):
            for b in range(1, self.bus_system + 1):
                model.addConstr(gp_agc_delta[b][t + 1] == gp_agc_delta[b][t] +  2 * math.pi * self.generator_sampling_time * (gp_agc_f[b][t + 1] - gp_agc_f[b][0]))
        
        # [AGC] reference setpoint for initial timeslot
        for b in self.generator_with_governors:
            model.addConstr(gp_agc_pr[b][0] == gp_agc_pg[b][0] * 0.005)

        # [AGC] reference setpoint for different generators        
        for b in self.generator_with_governors:
            for t in range(self.number_timeslots - 1):
                if t % self.agc_cycle == 0:
                    model.addConstr(gp_agc_pr[b][t] == gp_agc_pg[b][t] * 0.005)
                else:
                    model.addConstr(gp_agc_pr[b][t] == gp_agc_pr[b][t - 1])
    
        # [AGC] attacker's capability constraint of attackable buses
        for t in range(self.number_agc_steps):
            for b in range(1, self.bus_system + 1):
                model.addConstr( (bus_attackable[b] == 0) >> (av_load[b][t] == 0) )
                
        for b in range(self.bus_system + 1):
            if b not in self.attackable_buses:
                model.addConstr(bus_attackable[b] == 0)                
               
        
        # [AGC] k-resiliency constraint
        model.addConstr(num_attacked_buses == sum(bus_attackable) )
        model.addConstr(num_attacked_buses == 18 )

        ######################################################################
        ####################### attack constraints ###########################
        ######################################################################                         
        big_M = 10000
        for i in range(self.number_agc_steps):
            
            model.addConstr( num_attack_cycle >= (i + 1) - big_M * inference_gt[i] )
            model.addConstr( num_attack_cycle <= (i + 1) + big_M * ( 1- inference_gt[i]) )
            
            model.addConstr( num_attack_cycle >= (i - 1) - big_M * ( 1- inference_lt[i]) )
            model.addConstr( num_attack_cycle <= (i - 1) + big_M * inference_lt[i] )
            
            model.addGenConstrAnd(inference_attack_cycle[i], [inference_gt[i], inference_lt[i]])
        
        
        for i in range(self.number_agc_steps):
            for j in range(self.number_agc_steps):
                if i >= j:
                    model.addConstr( (inference_attack_cycle[i] == 1) >>  (agc_attackable[j] == 1) )
                else:
                    model.addConstr( (inference_attack_cycle[i] == 1) >>  (agc_attackable[j] == 0) )
            
        if type_defense == 'ADM':
            model.addConstr(num_attack_cycle == self.number_agc_steps - 1)
        elif type_defense =='BDD':
            model.addConstr(num_attack_cycle >= 1)
            model.addConstr(num_attack_cycle <= self.number_agc_steps - 1) 
        
        
        if attack_goal == 'UF':
            for t in range(self.number_agc_steps):
                or_ = []
                for b in self.generator_buses:    
                    model.addConstr(min_freqs[t] <= gp_gen_f[b][(t + 1) * self.agc_cycle - 2])
                    model.addConstr( (or_variables[t][b] == 1) >> (min_freqs[t] == gp_gen_f[b][(t + 1) * self.agc_cycle - 2]) )
                
                for b in range(self.bus_system + 1):
                    if b not in self.generator_buses:
                        model.addConstr(or_variables[t][b] == 0)
              
                model.addGenConstrOr(or_constraints[t] , or_variables[t])
                model.addConstr(or_constraints[t] == 1)
                
                model.addConstr( (inference_attack_cycle[t] == 1) >>  (min_freq == min_freqs[t] * 60) ) 

            model.addConstr(min_freq <= self.uf_threshold)
        
        elif attack_goal == 'OF':
            
            for t in range(self.number_agc_steps):
                or_ = []
                for b in self.generator_buses:    
                    model.addConstr(max_freqs[t] >= gp_gen_f[b][(t + 1) * self.agc_cycle - 2])
                    model.addConstr( (or_variables[t][b] == 1) >> (max_freqs[t] == gp_gen_f[b][(t + 1) * self.agc_cycle - 2]) )
                
                for b in range(self.bus_system + 1):
                    if b not in self.generator_buses:
                        model.addConstr(or_variables[t][b] == 0)
                
                model.addGenConstrOr(or_constraints[t] , or_variables[t])
                model.addConstr(or_constraints[t] == 1)
                
                model.addConstr( (inference_attack_cycle[t] == 1) >>  (max_freq == max_freqs[t] * 60) )            
            model.addConstr(max_freq >= self.of_threshold)
            

            
        if type_defense == 'ADM':
            model.setObjective(num_attacked_buses, gp.GRB.MINIMIZE)
        
        elif type_defense == 'BDD':
            model.setObjective(num_attacked_buses + num_attack_cycle, gp.GRB.MINIMIZE)
            
        model.optimize()
        
        self.execution_time = time.time() - start_time
        
        print("Execution time", time.time() - start_time)
        
        print("Model Status:", model.Status)
        if attack_goal == 'UF':
            print("Minimum Frequency:", min_freq.x)
        elif attack_goal == 'OF':
            print("Maximum Frequency:", max_freq.x)
        print("Number of attack cycle", num_attack_cycle.x)
        print("Number of attacked buses", num_attacked_buses.x)
                    
        self.attack_time  = num_attack_cycle.x
        self.number_const = model.NumConstrs
        
        print("Number of constraints", self.number_const)
        
        # Finding attack bus number
        attacked_bus = -1
        for b in self.generator_buses: 
            if int(or_variables[int(self.attack_time)][b].x) == 1:
                attacked_bus = b
                break
            
        print("Attacked bus", attacked_bus)
        
        dictionary = dict()
        dictionary['PL']    = [[gp_load[i][j].x for j in range(len(gp_load[0]))] for i in range(len(gp_load))] 
        dictionary['PL-A']   = [[gp_attacked_load[i][j].x for j in range(len(gp_attacked_load[0]))] for i in range(len(gp_attacked_load))] 
        dictionary['PG']    = [[gp_gen_pg[i][j].x for j in range(len(gp_gen_pg[0]))] for i in range(len(gp_gen_pg))]
        dictionary['PM']    = [[gp_gen_pm[i][j].x for j in range(len(gp_gen_pm[0]))] for i in range(len(gp_gen_pm))]
        dictionary['Delta'] = [[gp_gen_delta[i][j].x for j in range(len(gp_gen_delta[0]))] for i in range(len(gp_gen_delta))]
        dictionary['F']     = [[gp_gen_f[i][j].x for j in range(len(gp_gen_f[0]))] for i in range(len(gp_gen_f))]
        
        dictionary['AGC-PG']    = [[gp_agc_pg[i][j].x for j in range(len(gp_agc_pg[0]))] for i in range(len(gp_agc_pg))]
        dictionary['AGC-PM']    = [[gp_agc_pm[i][j].x for j in range(len(gp_agc_pm[0]))] for i in range(len(gp_agc_pm))]
        dictionary['AGC-PR']    = [[gp_agc_pr[i][j].x for j in range(len(gp_agc_pr[0]))] for i in range(len(gp_agc_pr))]
        dictionary['AGC-Delta'] = [[gp_agc_delta[i][j].x for j in range(len(gp_agc_delta[0]))] for i in range(len(gp_agc_delta))]
        dictionary['AGC-F']     = [[gp_agc_f[i][j].x for j in range(len(gp_agc_f[0]))] for i in range(len(gp_agc_f))]
        
        dictionary['Max-F'] = max_freq.x
        dictionary['C']     = model.NumConstrs
        dictionary['AT']    = self.attack_time
        dictionary['AB']    = int(attacked_bus)
        dictionary['T']     = self.execution_time
        
        with open('output-files/' + filename_to_save_output, "w") as outfile:
            json.dump(dictionary, outfile)    
            
        gp_load        = [[gp_load[i][j].x for j in range(len(gp_load[0]))] for i in range(len(gp_load))] 
        gp_gen_pg      = [[gp_gen_pg[i][j].x for j in range(len(gp_gen_pg[0]))] for i in range(len(gp_gen_pg))]
        gp_gen_pm      = [[gp_gen_pm[i][j].x for j in range(len(gp_gen_pm[0]))] for i in range(len(gp_gen_pm))]
        gp_gen_delta   = [[gp_gen_delta[i][j].x for j in range(len(gp_gen_delta[0]))] for i in range(len(gp_gen_delta))]
        gp_gen_f       = [[gp_gen_f[i][j].x for j in range(len(gp_gen_f[0]))] for i in range(len(gp_gen_f))]
        
        gp_agc_f       = [[gp_agc_f[i][j].x for j in range(len(gp_agc_f[0]))] for i in range(len(gp_agc_f))]
        gp_agc_pg      = [[gp_agc_pg[i][j].x for j in range(len(gp_agc_pg[0]))] for i in range(len(gp_agc_pg))]
        gp_agc_pr      = [[gp_agc_pr[i][j].x for j in range(len(gp_agc_pr[0]))] for i in range(len(gp_agc_pr))]
        gp_agc_load    =  [[gp_attacked_load[i][j].x for j in range(len(gp_attacked_load[0]))] for i in range(len(gp_attacked_load))] 
                    
        self.visualize("", gp_load, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{L}~\mathit{(p.u.)}$ ", self.attackable_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{L}_{' + str(b) + r'}$') for b in self.attackable_buses], 'upper right', '', 'plots/Case-Study-2_Load')
        self.visualize("", gp_gen_pg, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{G}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{G}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Generating-Power')
        self.visualize("", gp_gen_pm, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{M}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{M}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Mechanical-Power')
        self.visualize("", gp_agc_pr, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{P}^{R}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{P}^{R}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Reference-Power')
        self.visualize("", gp_gen_delta, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\delta~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\delta_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Rotor-Angle')
        self.visualize("", gp_gen_f, 60, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{F}~\mathit{(p.u.)}$ ", self.generator_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.generator_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')
        self.visualize("", gp_agc_load, 1, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\bar{\mathcal{P}}^L~\mathit{(p.u.)}$ ", self.attackable_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.attackable_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')
        self.visualize("", gp_agc_f, 60, r"$\mathcal{T}~(\frac{1}{60}~sec)$", r"$\mathcal{F}^\mathit{AGC}~\mathit{(p.u.)}$ ", self.pv_buses, self.number_timeslots - 1, [(r'$\mathcal{F}_{' + str(b) + r'}$') for b in self.pv_buses], 'upper right', '', 'plots/Case-Study-2_Frequency')              
        
        #for b in self.attackable_buses:
        #    for t in range(self.number_agc_steps):
        #        print(b, t, av_load[b][t].x)
        
        for b in self.attackable_buses:
            plt.figure()
            plt.xlabel('Load Measurement (P.U.) at Time, t - 1')
            plt.ylabel('Deviation Measurement (P.U.) at Time, t')
            plt.title('ADM for Bus' + str(b))
            for c in range(len(self.all_vertices[b])):
                plt.plot(self.all_vertices[b][c][:, 0], self.all_vertices[b][c][:, 1])
               
                for e in range(len(self.all_vertices[b][c])):
                    #print(b, c, e, self.all_vertices[b][c][e])
                    plt.scatter(self.all_vertices[b][c][:, 0], self.all_vertices[b][c][:, 1], s = 10)
        
            for t in range(1, self.number_agc_steps):
                plt.scatter(gp_agc_attacked_load[b][(t - 1)].x, gp_agc_attacked_load[b][t].x - gp_agc_attacked_load[b][t - 1].x,  s = 50)
        return int(num_attack_cycle.x + 1) * 60
        
    def get_min_load_timeslot(self):

        min_load_timeslot = -1
        min_load = 10
        bus = 39
        
        for t in range(len(self.benign_loads[bus])):
            if self.benign_loads[bus][t] < min_load:
                min_load          = self.benign_loads[bus][t]
                min_load_timeslot = t
        return min_load_timeslot, min_load
            
    def get_max_load_timeslot(self):
        
        max_load_timeslot = -1
        max_load = 0
        bus = 39
        
        for t in range(len(self.benign_loads[bus])):
            if self.benign_loads[bus][t] > max_load:
                max_load          = self.benign_loads[bus][t]
                max_load_timeslot = t
        return max_load_timeslot, max_load
            
    def get_closest_load(self, load_to_match):
        
        load_timeslot = -1
        diff = 10
        bus = 39
        
        for t in range(len(self.benign_loads[bus])):
            if abs(self.benign_loads[bus][t] - load_to_match) < diff:
                load          = self.benign_loads[bus][t]
                diff          = abs(self.benign_loads[bus][t] - load_to_match)
                load_timeslot = t
        return load_timeslot, load

if __name__ == "__main__":
    
    print("Grid Parameter Initialization")

    bus_system                  = 39
    number_days_load_data       = 30
    nominal_frequency           = 1
    agc_cycle                   = 60
    generator_sampling_time     = 1 / agc_cycle
    agc_sampling_time           = 1
    
    prefix_filename_dataset                = 'data/Cleaned-Load-Dataframe_'
    prefix_filename_bus_susceptance_matrix = 'data/Bus-Susceptance-Matrix_'
    prefix_filename_bus_parameters         = 'data/Bus-Parameters_'
    prefix_filename_generator_parameters   = 'data/Generator-Parameters_'
    
    generator_with_governors    = np.array([30, 31, 32, 33, 34, 35, 39])  # Generators with governors
    
    load_interval               = 600
    uf_threshold                = 59.5
    of_threshold                = 60.5
    
    
    analytics = MISGUIDE(bus_system, number_days_load_data, nominal_frequency, agc_cycle, generator_sampling_time, agc_sampling_time, uf_threshold, of_threshold)
    analytics.initialize_files(prefix_filename_dataset, prefix_filename_bus_susceptance_matrix, prefix_filename_bus_parameters, prefix_filename_generator_parameters)
    analytics.initialize_bus_parameters(generator_with_governors)
    analytics.initialize_generator_parameters()
    
    analytics.bdd_load_threshold(load_interval)
    analytics.adm_load_threshold()
    print()

    print("Benign System Analysis")

    type_analysis            = 'Benign'  # 'Benign', 'Synthesis', 'Optimal', 'Resiliency'
    attack_goal              = 'None'    # 'None', 'UF', 'OF'
    type_defense            = 'None'    # 'None', 'BDD', 'ADM'
    max_buses_to_attack      = 0
    num_agc_cycles_to_attack = 0
    number_timeslots         = 3000
    variable_load_timeslots  = 120
    filename_to_save_output  = str(type_analysis) + '_' + str(attack_goal) + '_' + str(type_defense) + '_Bus-' + str(max_buses_to_attack) + '.json'
    
    analytics.benign_system_analysis(number_timeslots, variable_load_timeslots, filename_to_save_output)
    print()