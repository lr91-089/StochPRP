# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:27:12 2025

@author: un_po
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import logging
from collections import deque,defaultdict
from itertools import combinations
import itertools
import networkx as nx
import threading
import random
import networkx as nx
import os
from datetime import datetime

import gurobipy as gp
from gurobipy import Model, GRB, quicksum, tuplelist
from tsp_heuristic_construction import VehicleRouteOptimizer
#from RCCPSymm import separate_Rounded_Capacity_cuts_symm
#from RCCPAsymm import separate_Rounded_Capacity_cuts_symm




gurobi_status_dict = {1: 'LOADED',
  2: 'OPTIMAL',
  3: 'INFEASIBLE',
  4: 'INF_OR_UNBD',
  5: 'UNBOUNDED',
  6: 'CUTOFF',
  7: 'ITERATION_LIMIT',
  8: 'NODE_LIMIT',
  9: 'TIME_LIMIT',
  10: 'SOLUTION_LIMIT',
  11: 'INTERRUPTED',
  12: 'NUMERIC',
  13: 'SUBOPTIMAL',
  14: 'INPROGRESS',
  15: 'USER_OBJ_LIMIT',
  16: 'WORK_LIMIT',
  17: 'MEM_LIMIT'}

def parse_mvprp_instance(file_path):
    data = {}
    suppliers = []
    retailers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the relevant section
    start_index = next(i for i, line in enumerate(lines) if "DIMENSION" in line)
    
    # Process relevant lines
    for line in lines[start_index:]:
        line = line.strip()
        if not line or line.startswith("COMMENT"):  # Skip empty lines and comments
            continue
        
        if "DIMENSION:" in line:
            data["num_nodes"] = int(line.split(":")[1].strip())
        elif "HORIZON:" in line:
            data["periods"] = int(line.split(":")[1].strip())
        elif "NUMBER OF VEHICLES:" in line:
            data["num_vehicles"] = int(line.split(":")[1].strip())
        elif "VEHICLE CAPACITY:" in line:
            data["Q"] = int(line.split(":")[1].strip())
        elif line.startswith("SUPPLIER"):
            continue  # Skip header
        elif line.startswith("0"):
            values = line.split()
            suppliers.append({
                "id": int(values[0]),
                "x": float(values[1]),
                "y": float(values[2]),
                "start_inv": float(values[3]),
                "inv_cost": float(values[4]),
                "var_prod_cost": float(values[5]),
                "setup_prod_cost": float(values[6]),
                "prod_capacity": float(values[7]),
                "max_inv": float(values[8])
            })
        elif line.startswith("RETAILER"):
            continue  # Skip header
        else:
            values = line.split()
            retailers.append({
                "id": int(values[0]),
                "x": float(values[1]),
                "y": float(values[2]),
                "start_inv": float(values[3]),
                "max_inv": int(values[4]),
                "min_inv": int(values[5]),
                "demand": int(values[6]),
                "inv_cost": float(values[7])
            })
    
    data["supplier"] = suppliers[0]
    data["retailers"] = retailers
    
    return data



def eliminate_subtours_components(model, edges,z, period, mipsol=True):
    #We don't care for the multiple edges, because we only check for the paths
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e[0],e[1],capacity=edges[e]["capacity"],vehicle=edges[e]["vehicle"])
    #G.add_edges_from(list(edges.keys()),edges)
    scc = [
        c
        for c in sorted(nx.strongly_connected_components(G), key=len) if len(c)>2
    ]
    for comp in scc:
        #if len(comp)>1:
            if 0 not in comp:
                comp_edges = list(combinations(comp,2))
                comp_edges += [(j,i) for i,j in comp_edges]
                subG = G.subgraph(comp)
                K = set(nx.get_edge_attributes(subG, "vehicle").values())
                if len(K)>1:
                    print("multiple vehicles in comp",K,len(comp))
                #print("st lazy cut added", len(comp))
                #glob_st_cuts.append(len(comp))
                zm = 0.0
                m = -1
                for i in list(comp):
                    if z[i]>zm:
                        m = i
                        zm = z[i]
                for k in K:
                    if mipsol==True:
                        model.cbLazy(
                            quicksum(model._x[i, j,period,k] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period,k] for i in comp)-model._z[m,period,k]
                        )
                    else:
                        model.cbCut(
                            quicksum(model._x[i, j,period,k] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period,k] for i in comp)-model._z[m,period,k]
                        )
                """
                all_x_vars = model.cbGetSolution(model._x)
                epsilon = pow(10, -4)
                x_arcs = {a:all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
                all_z_vars = model.cbGetSolution(model._z)
                all_k_vars = model.cbGetSolution(model._k)
                x_edges = {(a[0],a[1]):{"capacity":x_arcs[a],"vehicle":a[3]} for a in x_arcs if a[2]==1}
                z_arcs = {(i,t):all_z_vars[(i,t)] for i in nodes for t in periods if all_z_vars[(i,t)] > epsilon}
                k_dict = {t:all_k_vars[t] for t in periods if all_k_vars[t]>epsilon}
                for e in x_arcs:
                    if e[2]==1:
                        print(e,x_arcs[e])"""
                return True
    return False

def cb(model, where):
    def all_values_close_to_one(values, tol=0.01):
        """
        Checks if all values in the array are close to 1 within a given tolerance.
        
        Parameters:
            values (array-like): List or NumPy array of values between 0 and 1.
            tol (float): Tolerance level (default is 0.01).
            
        Returns:
            bool: True if all values are close to 1 within the tolerance, False otherwise.
        """
        #return False
        return np.all(np.isclose(values, 1, atol=tol))
    
    
           
                
                



def input_ordering(df, nodes,depot):
    def distance(city1, city2):
        c1 = coordinates[city1]
        c2 = coordinates[city2]
        diff = (c1[0]-c2[0], c1[1]-c2[1])
        return np.round(math.sqrt(diff[0]*diff[0]+diff[1]*diff[1]),0)
    df["points"] = df[["x","y"]].values.tolist()
    
    coordinates = df.set_index("id")["points"].to_dict()
    coordinates[0] = [depot["x"],depot["y"]]
    dist = {(c1, c2): distance(c1, c2) for c1 in nodes for c2 in nodes if c1!=c2}
    max_dist = max(dist[0,i] for i in nodes if i>0)
    df["dist_col_norm"] = [dist[0,i]/max_dist for i in nodes if i>0]
    df["demand_norm"] = df["demand"]/df["demand"].max()
    df["order_col"] = df["dist_col_norm"]*df["demand_norm"]
    df = df.sort_values("order_col",ascending=False)
    df["id"] = range(1,df.shape[0]+1)
    df = df.reset_index(drop=True)
    df = df.drop(["dist_col_norm","demand_norm","order_col"], axis=1)
    return df

def input_ordering_demand(df, nodes):
    df = df.sort_values("demand",ascending=False)
    df["id"] = range(1,df.shape[0]+1)
    df = df.reset_index(drop=True)
    return df


def create_cp_mip(file_path,recourse_gamma = 3, stoch=True):
    print(file_path)
    dataFile = parse_mvprp_instance(file_path)
    
    
    
    num_nodes,num_vehicles, num_days, Q = dataFile["num_nodes"],dataFile["num_vehicles"],dataFile["periods"],dataFile["Q"]
    
    depot = dataFile["supplier"]
    
    # Read remaining rows as customers
    customers = dataFile["retailers"]
        
    # List of nodes including the depot node 1
    nodes = [*range(num_nodes)]
    
    # Convert to DataFrame
    customers_df = pd.DataFrame(customers)
    
    customers_df = input_ordering_demand(customers_df, nodes)
    
    # Display parsed data
    print("General Information:")
    print(f"Number of nodes: {num_nodes}, Days Considered: {num_days}, Transport Capacity: {Q}\n")
    print("Depot Information:")
    print(depot, "\n")
    print("Customers DataFrame:")
    print(customers_df)
    
    Q = Q
    
    customers_df["points"] = customers_df[["x","y"]].values.tolist()
    
    coordinates = customers_df.set_index("id")["points"].to_dict()
    coordinates[0] = [depot["x"],depot["y"]]
    
    holding_costs = customers_df.set_index("id")["inv_cost"].to_dict()
    holding_costs[0] = depot["inv_cost"]
    
    
    
    
    customers = [*range(1,num_nodes)]
    
    periods = [*range(1,num_days+1)]
    periodsI = [*range(0,num_days+1)]
    
        
    daily_demand = {(i,t):customers_df.set_index("id")["demand"].to_dict()[i] for i in customers for t in periods}
    
    probabilities = {}

    if stoch == True:
        if num_vehicles == 2:
            vehicles = [num_vehicles-2,num_vehicles-1,num_vehicles,num_vehicles+1,num_vehicles+2]
            probabilities = {num_vehicles-2:0.1,num_vehicles-1:0.2,num_vehicles:0.4,num_vehicles+1:0.2,num_vehicles+2:0.1}
        else:
            probabilities = {num_vehicles-2:0.1,num_vehicles-1:0.2,num_vehicles:0.4,num_vehicles+1:0.2,num_vehicles+2:0.1}
            vehicles_1 = []
            for i in range(1,num_vehicles-2):
                vehicles_1.append(i)
                probabilities[i]=0.1
            vehicles = vehicles_1+[num_vehicles-2,num_vehicles-1,num_vehicles,num_vehicles+1,num_vehicles+2]
            
        
        det_vehicles = num_vehicles
        num_vehicles = num_vehicles+2
    else:
        det_vehicles = num_vehicles
        num_vehicles = num_vehicles+2
        vehicles = [*range(1,num_vehicles+1)]
    
    initial_inv = customers_df.set_index("id")["start_inv"].to_dict()
    initial_inv[0] = depot["start_inv"] 
    inv_cap = customers_df.set_index("id")["max_inv"].to_dict()
    inv_cap[0] = depot["max_inv"]
    
    
    production_capacity = {t:depot["prod_capacity"] for t in periods}
    
    var_prod_cost = depot["var_prod_cost"]
    
    setup_prod_cost = depot["setup_prod_cost"]
    
    
    

    
    def distance(city1, city2):
        c1 = coordinates[city1]
        c2 = coordinates[city2]
        diff = (c1[0]-c2[0], c1[1]-c2[1])
        return np.round(math.sqrt(diff[0]*diff[0]+diff[1]*diff[1]),0)
    
    dist = {(c1, c2): distance(c1, c2) for c1 in nodes for c2 in nodes if c1!=c2}
    
    
    #recourse costs
    #recourse costs
    
    r = {}
    for i in customers:
        r[i] = dist[0,i]*recourse_gamma
    if stoch == True:
        c = {}
        for k in vehicles:
            for i in nodes:
                for j in nodes:
                    if i!=j:
                        if j>0:
                            if k>0:
                                if det_vehicles==4 and k==1:
                                    c[i,j,k] = distance(i,j)*sum(probabilities[l] for l in range(k+1,num_vehicles+1))+r[j]*(1-sum(probabilities[l] for l in range(k+1,num_vehicles+1)))
                                else:
                                    c[i,j,k] = distance(i,j)*sum(probabilities[l] for l in range(k,num_vehicles+1))+r[j]*(1-sum(probabilities[l] for l in range(k,num_vehicles+1)))
                            else:
                                c[i,j,k] = r[j]
                        else:
                            if k>0:
                                if det_vehicles==4 and k==1:
                                    c[i,j,k] = distance(i,j)*sum(probabilities[l] for l in range(k+1,num_vehicles+1))
                                else:
                                    c[i,j,k] = distance(i,j)*sum(probabilities[l] for l in range(k,num_vehicles+1))
                            else:
                                c[i,j,k] = 0
    else:
        c = {(c1, c2,k): distance(c1, c2) for c1 in nodes for c2 in nodes if c1!=c2 for k in vehicles}
    
    
    t1 = num_days
    for t in periods:
        t_temp = sum(max(0,sum(daily_demand[i,j]-initial_inv[i] for j in range(t,num_days+1))-initial_inv[0]) for i in customers)
        if t_temp>0:
            if t_temp<t1:
                t1 = t_temp
    print(t1)
    
    t2i = {}
    for i in customers:
        t_i = num_days
        for t in periods:
            t_temp = max(sum(daily_demand[i,j]-initial_inv[i] for j in range(1,t+1)),0)
            if t_temp>0:
                if t_temp<t_i:
                    t_i = t_temp
        t2i[i] = t_i
    t2 = min(t2i.values())

    kt = sum(max(0,sum(daily_demand[i,j]-initial_inv[i] for j in range(1,t2+1))) for i in customers)    
    
    # Create the model object m
    env = gp.Env()
    
    if stoch==True:
        model_type = "stoch"
    else:
        model_type = "determ"
    model_name = f'model2_HPRP_symmBreakNew3_InputOrderingDemand_USerCuts_newSymmetryBreaking_{model_type}'
    m = gp.Model(model_name, env=env)
    
    # Decision variables: 
    
    # Edge variables = 1, if customer i is visited after h in period t by vehicke k
    var_x = m.addVars(dist, periods,vehicles, vtype=GRB.BINARY, name='x')
    #binary variable = 1 if setup occurs in period t
    var_y = m.addVars(periods, vtype=GRB.BINARY, name='y')
    #quantity produced in period t
    var_p = m.addVars(periods,lb=0.0, vtype=GRB.CONTINUOUS, name='p')
    #quantity of inventory at node i in period t
    var_I = m.addVars(nodes,periodsI,lb=0.0, vtype=GRB.CONTINUOUS, name='I')
    #quantity delivered to cust i in period t
    var_q = m.addVars(nodes,periods, vehicles,lb=0.0, vtype=GRB.CONTINUOUS, name='q')
    #binary variable equal to 1 if node i in N is visited in period t
    var_z = m.addVars(nodes,periods, vehicles, vtype=GRB.BINARY, name='z')
    #number of vehicles that leave the production plant in period t
    #var_k = m.addVars(periods,lb=0.0,ub=len(vehicles), vtype=GRB.CONTINUOUS, name='k')
    var_k = m.addVars(periods,vehicles, vtype=GRB.BINARY, name='k')
    
    
    
    m.modelSense = GRB.MINIMIZE
    fixed_production_costs = quicksum(setup_prod_cost*var_y[t] for t in periods)
    variable_production_costs = quicksum(var_prod_cost*var_p[t] for t in periods)
    inventory_holding_costs = quicksum(holding_costs[i]*var_I[i,t] for i in nodes for t in periods)
    routing_cost_approximation = quicksum(2*c[0,i,k] * var_z[i,t,k] for i in customers for t in periods for k in vehicles)
    m.setObjective(fixed_production_costs +variable_production_costs+inventory_holding_costs+routing_cost_approximation)
    
    m.addConstrs((var_I[0,t-1]+var_p[t] == var_I[0,t]+quicksum(var_q[i,t,k] for i in nodes for k in vehicles) for t in periods), name="Inventory balance at plant")
    m.addConstrs((var_I[i,t-1]+quicksum(var_q[i,t,k] for k in vehicles) == var_I[i,t]+daily_demand[i,t] for i in customers for t in periods), name="Inventory balance at customers")
    m.addConstrs((var_I[0,t] <= inv_cap[0] for t in periods), name="Inventory balance at plant")
    m.addConstrs((var_I[i,t-1]+quicksum(var_q[i,t,k] for k in vehicles) <= inv_cap[i] for i in customers for t in periods), name="Inventory capacity at client after delivery")
    m.addConstrs((var_p[t] <= min(production_capacity[t],sum(daily_demand[i,l] for i in customers for l in periods[t-1:]))*var_y[t] for t in periods), name="Production capacity at plant")
    m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],Q,sum(daily_demand[i,l] for l in periods[t-1:]))*var_z[i,t,k] for i in customers for t in periods for k in vehicles if k!=0), name="Min quantity delivered only if customer is visited in same period")
    if stoch==True:
        m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],sum(daily_demand[i,l] for l in periods[t-1:]))*var_z[i,t,k] for i in customers for t in periods for k in vehicles if k==0), name="Min quantity delivered only if customer is visited in same period Uncapacitated vehicle")
    #m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i) == var_z[i,t,k] for i in customers for t in periods for k in vehicles), name="customer visit link")
    m.addConstrs((quicksum(var_z[i,t,k] for k in vehicles)<=1 for i in customers for t in periods), name="customer visit at most once")
    #m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i)+quicksum(var_x[j,i,t,k] for j in nodes if j!=i) == 2*var_z[i,t,k] for i in nodes for t in periods for k in vehicles), name="degree constraints at client")
    #m.addConstrs((quicksum(var_x[1,j,t,k] for j in customers)+quicksum(var_x[j,1,t,k] for j in customers) == 2*var_k[t,k] for t in periods for k in vehicles), name="degree constraints at depot")
    #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers )  quicksum(var_x[j,0,t,k] for j in customers) == 2*var_z[0,t,k] for t in periods for k in vehicles), name="leave at most once")
    #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers) == for t in periods for k in vehicles), name="degree constraints at depot")
    
    m.addConstrs((quicksum(var_q[i,t,k] for i in customers) <= Q*var_z[0,t,k] for k in vehicles if k!=0 for t in periods), name="Capacity Constraints")
    
    #if stoch==False:
        #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)  <= quicksum(var_x[0,j,t,k-1] for j in customers)  for t in periods for k in vehicles[1:]), name="vehicle_symmetry")

    #m.addConstrs((quicksum(var_x[1,j,t,k] for j in customers)== var_k[t,k] for t in periods for k in vehicles), name="degree constraints at depot 1")
    #m.addConstrs((quicksum(var_x[j,1,t,k] for j in customers)== var_k[t,k] for t in periods for k in vehicles), name="degree constraints at depot 2")
    #m.addConstrs((var_z[i,t,k] <= var_k[t,k]  for t in periods for i in customers for k in vehicles), name="Customer only visited in period t if at least one vehicle leaves the plant")
    
    m.addConstrs((var_z[i,t,k]   <= var_z[0,t,k] for i in customers for t in periods for k in vehicles), name="vehicle_symmetry_LeavingDepot")

    if stoch==True and  det_vehicles==2:
        
        #det_vehicles-1
        m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[2:]), name="vehicle_symmetry1")
        m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k-1] for l in customers if l<i) for i in customers for t in periods for k in vehicles[2:det_vehicles-1]), name="vehicle_symmetry2")
        #m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,kl] for l in customers if l<i for kl in [1,2]) for i in customers[1:] for t in periods for k in [det_vehicles-1]), name="vehicle_symmetry2")
        #m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k] for l in customers if l<i) for i in customers[1:] for t in periods for k in vehicles[det_vehicles:]), name="vehicle_symmetry2")

    else:
        if stoch==False:
            m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k-1] for l in customers if l<i) for i in customers[1:] for t in periods for k in vehicles[1:]), name="vehicle_symmetry2")
        m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[1:]), name="vehicle_symmetry1")

        
    #m.addConstrs((var_x[i,j,t,k]+var_x[j,i,t,k]  <= var_z[i,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles), name="subtours of size 2")
    
    #m.addConstrs((quicksum(i*var_x[0,i,t,k] for i in customers)  <= quicksum(i*var_x[i,0,t,k] for i in customers)  for t in periods for k in vehicles), name="symmetric_cost_vi")

    
    """
    m.addConstrs((quicksum(var_x[1,j,t,k] for j in customers)<= 1 for t in periods for k in vehicles), name="degree constraints at depot 0")
    m.addConstrs((quicksum(var_x[1,j,t,k] for j in customers for k in vehicles)== var_k[t] for t in periods), name="degree constraints at depot 1")
    m.addConstrs((quicksum(var_x[j,1,t,k] for j in customers for k in vehicles) == quicksum(var_x[1,j,t,k] for j in customers for k in vehicles) for t in periods), name="degree constraints at depot 2")
    m.addConstrs((var_z[i,t] <= var_k[t]  for t in periods for i in customers), name="Customer only visited in period t if at least one vehicle leaves the plant")
    """
    #initialize initial inventory
    m.addConstrs(var_I[i,0]==initial_inv[i] for i in nodes)
    
    #Valid in equalities
    """
    m.addConstr(quicksum(var_y[t] for t in range(1,t1+1))>= 1)
    m.addConstr(quicksum(var_z[0,t,k] for k in vehicles for t in range(1,t2+1))>= float(np.ceil(kt/Q)))
    
    m.addConstrs((var_I[i,t-s-1] >= quicksum(daily_demand[i,t-j] for j in range(0,s+1))*(1-quicksum(var_z[i,t-j,k] for k in vehicles for j in range(0,s+1))) for i in customers for t in periods for s in range(0,t)), name="Inventory inequality")

    m.addConstrs((var_z[i,t,k] <=var_z[0,t,k] for i in customers for t in periods for k in vehicles ), name="routing inequality")
    m.addConstrs((var_x[i,j,t,k] <=var_z[i,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles ), name="routing inequality x")
    m.addConstrs((var_x[i,j,t,k] <=var_z[j,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles ), name="routing inequality x")
    """
    
    
    m._x = var_x
    m._z = var_z
    m._q = var_q
    m._p = var_p
    m._y = var_y
    m._I = var_I
    m._Q = Q
    m._dist = dist
    m._c = c
    m._r = r
    m._nodes = nodes
    m._num_nodes = num_nodes
    m._customers = customers
    m._periods = periods
    m._vehicles = vehicles
    m._probabilities = probabilities
    m._benders_cuts = []
    m._theta_scale = {(t,s):None for s in probabilities for t in periods}
    m._found_sol = False
    m._fixed_production_costs = fixed_production_costs
    m._variable_production_costs = variable_production_costs
    m._inventory_holding_costs = inventory_holding_costs
    m._routing_costs = routing_cost_approximation
    m._cb_lastnode = 0
    #m.write("model.lp")
    m.params.LazyConstraints = 1
    m.params.TimeLimit = 10
    m.params.Threads = 4
    m._cb_last_lower_bound = 0.0
    m._cb_last_obj = np.inf
    return m,env


def generate_initial_solution(file_path, recourse_gamma,stoch=True):
         m,model_env = create_cp_mip(file_path,recourse_gamma=recourse_gamma,stoch = stoch)
         m.optimize(callback=cb)
         for iter2 in range(0):
                 m.setParam(GRB.Param.SolutionNumber, iter2)
                 print('%g ' % m.PoolObjVal, end='\n')
                 for v in m.getVars():
                      if v.xn > 1e-5:
                            #print ('%s %g' % (v.varName, v.xn))
                            print ('%s %g' % (v.varName, v.xn))
                 print("\n")
         error = ""
         used_vehicles = set()
         prod_quantity = set()
         cost_values = []
         costs_vars = [m._fixed_production_costs,m._variable_production_costs,m._inventory_holding_costs,m._routing_costs]
         solution = [e for e in m._z if m._z[e].x>0.5 and e[0]>0]
         route_optimizer = VehicleRouteOptimizer(m._c)
         # Generate routes using Christofides
         print("=== CHRISTOFIDES ALGORITHM ===")
         christofides_routes = route_optimizer.generate_all_routes(
             solution, 
             method='christofides', 
             improve_with_2opt=True
         )
         
         print("\n" + "="*50)
         """
         # Generate routes using Nearest Neighbor (faster heuristic)
         #print("=== NEAREST NEIGHBOR HEURISTIC ===")
         
         nn_routes = route_optimizer.generate_all_routes(
             solution, 
             method='nearest_neighbor', 
             improve_with_2opt=True
         )
         total_nn = sum(distance for _, distance in nn_routes.values())
         print(f"Total distance (Nearest Neighbor): {total_nn:.2f}")
         print(f"Difference: {abs(total_christofides - total_nn):.2f}")
         print(f"NN vs Christofides: {(total_nn/total_christofides - 1)*100:.1f}% difference")
         """
         # Compare results
         """
         print("\n=== COMPARISON ===")
         total_christofides = sum(distance for _, distance in christofides_routes.values())
                
         
         print(f"Total distance (Christofides): {total_christofides:.2f}")"""
         """
         if m.status==2 or m.SolCount>0:
             for var in costs_vars:
                 print(var.getValue())
                 cost_values.append(var.getValue())
             
             for iter2 in range(1):
                     m.setParam(GRB.Param.SolutionNumber, iter2)
                     print('%g ' % m.PoolObjVal, end='\n')
                     for v in m.getVars():
                          if v.xn > 1e-5:
                                #print ('%s %g' % (v.varName, v.xn))
                                print ('%s %g' % (v.varName, v.xn))
                     print("\n")
             print("\n")
             for t in m._periods:
                 for k in m._vehicles:
                     x_arcs = [(e[0],e[1]) for e in m._x if m._x[e].x>0.5 if e[2]==t and e[3]==k]
                     q_arcs = [e for e in m._q if m._q[e].x>0.5 if e[1]==t]
                     z_arcs = [e[0] for e in m._z if m._z[e].x>0.5 if e[1]==t and e[2]==k]
                     load = sum(m._q[e].x for e in q_arcs if e[0] in z_arcs)
                     
         else:
             for var in costs_vars:
                 cost_values.append(-1)
         obj = None
         try:
             obj = m.ObjVal
         except:
             obj = None
             
         for i in range(3):
             cost_values.append(-1)
         """
         
         #cost_labels = ["fixed_prod_cost","variable_prod_cost","inv_hold_cost","routing_cost","recourse_cost","unused_routing_cost","expected_cost_diff"]
         #results.append([file_path.split("/")[-1].strip(".dat"),m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime,sorted(used_vehicles),m._Q,error,recourse_gamma]+cost_values)
         #df = pd.DataFrame(results, columns=["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recoursegamma"]+cost_labels)
         #df.to_csv("results/construction_procedure.csv")
         return m,christofides_routes

def run_cp_mip():
    mainFolderPath = "./Instances/MVPRP_DatasetI1"
    folder = os.fsencode(mainFolderPath)
    filenames = []
    for subdir, dirs, files in os.walk(mainFolderPath):
        for file in files:
            #if "_V4_" in file:
                experiment_folder = subdir.split(os.sep)[-1]
                filepath = os.path.join(subdir, file)
                filenames.append(filepath)
    filenames =["./Instances/Data_Test/MVPRP_C10_P3_V2_I1.dat"]
    #filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V4_I1.dat","./Instances/MVPRP_DatasetI1/MVPRP_C40_P6_V4_I1.dat"]
    results = []
    recourse_gamma=3
    for file_path in filenames:
        m,model_env = create_cp_mip(file_path,recourse_gamma=recourse_gamma,stoch = True)
        m.optimize(callback=cb)
        error = ""
        used_vehicles = set()
        prod_quantity = set()
        cost_values = []
        costs_vars = [m._fixed_production_costs,m._variable_production_costs,m._inventory_holding_costs,m._routing_costs]
        solution = [e for e in m._z if m._z[e].x>0.5 and e[0]>0]
        route_optimizer = VehicleRouteOptimizer(m._c)
        # Generate routes using Christofides
        print("=== CHRISTOFIDES ALGORITHM ===")
        christofides_routes = route_optimizer.generate_all_routes(
            solution, 
            method='christofides', 
            improve_with_2opt=True
        )
        
        print("\n" + "="*50)
        """
        # Generate routes using Nearest Neighbor (faster heuristic)
        #print("=== NEAREST NEIGHBOR HEURISTIC ===")
        
        nn_routes = route_optimizer.generate_all_routes(
            solution, 
            method='nearest_neighbor', 
            improve_with_2opt=True
        )
        total_nn = sum(distance for _, distance in nn_routes.values())
        print(f"Total distance (Nearest Neighbor): {total_nn:.2f}")
        print(f"Difference: {abs(total_christofides - total_nn):.2f}")
        print(f"NN vs Christofides: {(total_nn/total_christofides - 1)*100:.1f}% difference")
        """
        # Compare results
        print("\n=== COMPARISON ===")
        total_christofides = sum(distance for _, distance in christofides_routes.values())
               
        
        print(f"Total distance (Christofides): {total_christofides:.2f}")
        if m.status==2 or m.SolCount>0:
            for var in costs_vars:
                print(var.getValue())
                cost_values.append(var.getValue())
            
            for iter2 in range(1):
                    m.setParam(GRB.Param.SolutionNumber, iter2)
                    print('%g ' % m.PoolObjVal, end='\n')
                    for v in m.getVars():
                         if v.xn > 1e-5:
                               #print ('%s %g' % (v.varName, v.xn))
                               print ('%s %g' % (v.varName, v.xn))
                    print("\n")
            print("\n")
            for t in m._periods:
                for k in m._vehicles:
                    x_arcs = [(e[0],e[1]) for e in m._x if m._x[e].x>0.5 if e[2]==t and e[3]==k]
                    q_arcs = [e for e in m._q if m._q[e].x>0.5 if e[1]==t]
                    z_arcs = [e[0] for e in m._z if m._z[e].x>0.5 if e[1]==t and e[2]==k]
                    load = sum(m._q[e].x for e in q_arcs if e[0] in z_arcs)
                    
        else:
            for var in costs_vars:
                cost_values.append(-1)
        obj = None
        try:
            obj = m.ObjVal
        except:
            obj = None
            
        for i in range(3):
            cost_values.append(-1)
       
        
        cost_labels = ["fixed_prod_cost","variable_prod_cost","inv_hold_cost","routing_cost","recourse_cost","unused_routing_cost","expected_cost_diff"]
        results.append([file_path.split("/")[-1].strip(".dat"),m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime,sorted(used_vehicles),m._Q,error,recourse_gamma]+cost_values)
        df = pd.DataFrame(results, columns=["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recoursegamma"]+cost_labels)
        df.to_csv("results/construction_procedure.csv")

#run_cp_mip()


