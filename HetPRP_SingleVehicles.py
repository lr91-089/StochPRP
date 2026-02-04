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
import traceback
from datetime import datetime

import gurobipy as gp
from gurobipy import Model, GRB, quicksum, tuplelist
from construction_procedure import generate_initial_solution
from cap_sep_mip import separate_fractional_capacity_inequalities

#from RCCPSymm import separate_Rounded_Capacity_cuts_symm
#from RCCPAsymm import separate_Rounded_Capacity_cuts_symm


stoch = True
MIPstart = True
z_small = False
calc_root = False
VARIABLE_COST = False
UB = False
VI = False

if UB==True:
    from UB_deterministic_model import solve_ub_model

class SolverState:
    def __init__(self):
        self.model_ub = np.inf
        self.new_solution = {"x":{}}
        self.master_finished = False
        self.found_new = False
        self.mutex = threading.Lock()

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
                    if z_small == False:
                        model.cbLazy(
                            quicksum(model._x[i, j,period,k] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period,k] for i in comp)-model._z[m,period,k]
                        )
                    else:
                        model.cbLazy(
                            quicksum(model._x[i, j,period,k] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period] for i in comp)-model._z[m,period]
                        )
                    """
                    if mipsol==True:
                        model.cbLazy(
                            quicksum(model._x[i, j,period,k] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period,k] for i in comp)-model._z[m,period,k]
                        )
                    else:
                        model.cbCut(
                            quicksum(model._x[i, j,period,k] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period,k] for i in comp)-model._z[m,period,k]
                        )"""
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


def order_solution(model, x):
    
    sol_arcs = set()
    customer_fin = set()
    
    # Convert x to a more accessible format: (i,j,t) -> value
    arc_dict = {(i, j, t): v for (i, j, t), v in x.items() if v > 0.5}
    
    processed_arcs = set()
    
    for t in model._periods:
        vehicle_id = 1
        base_routes = {}
        route_exp_cost = {}
        
        # Find all tours starting from depot (node 0) in period t
        depot_starts = [(i, j, period) for (i, j, period) in arc_dict.keys() 
                       if i == 0 and period == t and (i, j, period) not in processed_arcs]
        
        for start_arc in depot_starts:
            if start_arc in processed_arcs:
                continue
                
            # Build complete tour from this starting arc
            tour = []
            current_arc = start_arc
            current_node = start_arc[1]  # Start from the destination of the first arc
            
            # Add the starting arc
            tour.append((current_arc[0], current_arc[1]))
            processed_arcs.add(current_arc)
            customers = {current_node} if current_node != 0 else set()
            
            # Follow the tour until we return to depot
            while current_node != 0:
                # Find next arc from current_node in the same period
                next_arc = None
                for (i, j, period) in arc_dict.keys():
                    if (i == current_node and period == t and 
                        (i, j, period) not in processed_arcs):
                        next_arc = (i, j, period)
                        break
                
                if next_arc is None:
                    print(f"Warning: Could not complete tour from node {current_node} in period {t}")
                    break
                
                # Add arc to tour
                tour.append((next_arc[0], next_arc[1]))
                processed_arcs.add(next_arc)
                current_node = next_arc[1]
                if current_node != 0:
                    customers.add(current_node)
            
            # Only store complete tours (that return to depot)
            if current_node == 0 and tour:
                route_cost = sum(model._dist[(i, j)] for (i, j) in tour)
                recourse_penalty = sum(model._r[i] for i in customers if i > 0)
                base_routes[vehicle_id] = tour
                route_exp_cost[vehicle_id] = recourse_penalty - route_cost
                vehicle_id += 1
        
        # Sort routes by expected cost (highest first)
        if route_exp_cost:
            sorted_vehicles = sorted(route_exp_cost.keys(), key=lambda k: route_exp_cost[k], reverse=True)
            
            # Special check for model._med == 4
            if model._med == 4 and len(sorted_vehicles) >= 2:
                # Create a temporary customer_fin to check the condition
                temp_customer_fin = set()
                
                # Add customers for vehicle 1 (k=1)
                for (i, j) in base_routes[sorted_vehicles[0]]:
                    if j > 0:
                        temp_customer_fin.add((j, t, 1))
                
                # Add customers for vehicle 2 (k=2) 
                for (i, j) in base_routes[sorted_vehicles[1]]:
                    if j > 0:
                        temp_customer_fin.add((j, t, 2))
                
                # Check condition for vehicle 2 (k=2)
                condition_holds = True
                for i in model._customers:
                    # Check if customer i is served by vehicle 2
                    if (i, t, 2) in temp_customer_fin:
                        # Sum customers served by vehicle 1 with index < i
                        sum_prev = sum(1 for l in model._customers 
                                     if l < i and (l, t, 1) in temp_customer_fin)
                        
                        # The condition: customer_fin(i,t,2) <= sum of customer_fin(l,t,1) for l < i
                        # Since customer_fin is binary (0 or 1), this becomes: 1 <= sum_prev
                        if sum_prev == 0:  # Condition doesn't hold
                            condition_holds = False
                            break
                
                # If condition doesn't hold, swap vehicles 1 and 2
                if not condition_holds:
                    sorted_vehicles[0], sorted_vehicles[1] = sorted_vehicles[1], sorted_vehicles[0]
            for new_id, old_id in enumerate(sorted_vehicles, 1):
                for (i, j) in base_routes[old_id]:
                    sol_arcs.add((i, j, t, new_id))
                    if j > 0:  # Don't add depot to customer_fin
                        customer_fin.add((j, t, new_id))
    return sol_arcs, customer_fin

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
    
    ###MIPSOL callback
    if where == GRB.Callback.MIPSOL:
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        UB = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        LB = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        gap = (UB-LB)/UB
        all_x_vars = model.cbGetSolution(model._x)
        all_z_vars = model.cbGetSolution(model._z)
        epsilon = pow(10, -4)
        x_arcs = {a:all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
        #subtours
        for t in random.sample(model._periods,len(model._periods)):
            z_vars = {a[0]:all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1]==t)}
            #we need subtours for zero vehicle too here!
            x_arcs_dict = {(a[0],a[1]):{"capacity":x_arcs[a],"vehicle":a[3]} for a in x_arcs if a[2]==t}
            if eliminate_subtours_components(model, x_arcs_dict,z_vars,t)==True:
                return True
            
            
            
    
    #"""
    if where == GRB.Callback.MIPNODE:
        cur_obj = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)
        sol_cnt =  model.cbGet(gp.GRB.Callback.MIPNODE_SOLCNT)
        
        if model._solver_state.found_new == True:
            if sol_cnt>0:
                if np.round(cur_obj,4) > np.round(model._solver_state.model_ub,4)+pow(10,-4):
                    with model._solver_state.mutex:
                        temp_sol = model._solver_state.new_solution
                    ordered_x_arcs, customers_visited = order_solution(model,temp_sol["x"])
    
                    # Create solution values for all x variables
                    new_vars = []
                    new_vals = []
                    
                    
                    # Iterate through all x variables
                    for (i, j) in model._dist:
                        for t in model._periods:
                            for k in model._vehicles:
                                new_vars.append(model._x[i, j, t, k])
                                # Set to 1 if this tuple is in the solution, 0 otherwise
                                if (i, j, t, k) in ordered_x_arcs:
                                    new_vals.append(1.0)
                                else:
                                    new_vals.append(0.0)
                    """
                    for i in model._nodes:
                        for t in model._periods:
                            for k in model._vehicles:
                                new_vars.append(model._z[i, t, k])
                                if (i, t, k) in customers_visited :
                                    new_vals.append(1.0)
                                else:
                                    new_vals.append(0.0)
                    
                    new_vars += [model._I[var] for var in temp_sol["I"]]
                    new_vars += [model._y[var] for var in temp_sol["y"]]
                    new_vars += [model._p[var] for var in temp_sol["p"]]
                    new_vals += list(temp_sol["I"].values())
                    new_vals += list(temp_sol["y"].values())
                    new_vals += list(temp_sol["p"].values()) """
                    try_sol = model.cbSetSolution(new_vars, new_vals)
                    try_sol = model.cbUseSolution()
                    print("new solution:", model._solver_state.model_ub, cur_obj,try_sol)
                    #if try_sol==1e+100:
                     #   breakpoint()
                      #  print("not used")
                    model._solver_state.found_new = False
            else:
                if model._solver_state.model_ub<np.inf:
                    with model._solver_state.mutex:
                        temp_sol = model._solver_state.new_solution
                    ordered_x_arcs, customers_visited = order_solution(model,temp_sol["x"])
    
                    # Create solution values for all x variables
                    new_vars = []
                    new_vals = []
                    
                   #breakpoint()
                    # Iterate through all x variables
                    for (i, j) in model._dist:
                        for t in model._periods:
                            for k in model._vehicles:
                                new_vars.append(model._x[i, j, t, k])
                                # Set to 1 if this tuple is in the solution, 0 otherwise
                                if (i, j, t, k) in ordered_x_arcs:
                                    new_vals.append(1.0)
                                else:
                                    new_vals.append(0.0)
                    """
                    for i in model._nodes:
                        for t in model._periods:
                            for k in model._vehicles:
                                new_vars.append(model._z[i, t, k])
                                if (i, t, k) in customers_visited :
                                    new_vals.append(1.0)
                                else:
                                    new_vals.append(0.0)
                    
                    new_vars += [model._I[var] for var in temp_sol["I"]]
                    new_vars += [model._y[var] for var in temp_sol["y"]]
                    new_vars += [model._p[var] for var in temp_sol["p"]]
                    new_vals += list(temp_sol["I"].values())
                    new_vals += list(temp_sol["y"].values())
                    new_vals += list(temp_sol["p"].values()) """
                    try_sol = model.cbSetSolution(new_vars, new_vals)
                    try_sol = model.cbUseSolution()
                    print("new solution:", model._solver_state.model_ub, cur_obj,try_sol)
                    model._solver_state.found_new = False
        epsilon = pow(10, -4)
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            cur_lb =  model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            cur_ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
            #if nodecnt<1:
            ncut = 0
            if (nodecnt<200 or (nodecnt>200 and nodecnt - model._cb_lastnode >= 400)):
                model._cb_lastnode = nodecnt
                if model._cb_last_lower_bound<cur_lb-epsilon or model._cb_last_ub-epsilon>cur_ub:
                    all_x_vars = model.cbGetNodeRel(model._x)
                    all_z_vars = model.cbGetNodeRel(model._z)
                    all_q_vars = model.cbGetNodeRel(model._q)
                    epsilon = pow(10, -4)
                    x_arcs = {a:all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
                    #if all_values_close_to_one(list(x_arcs.values()))==False:
                    for t in random.sample(model._periods,len(model._periods)):
                        for k in model._vehicles:
                            if k>0:
                                x_arcs_dict = {(a[0],a[1]):{"capacity":x_arcs[a],"vehicle":a[3]} for a in x_arcs if (a[2]==t and a[3]==k)}
                                if z_small == False:
                                    z_vars = {a[0]:all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1]==t and a[2]==k)}
                                else:
                                    z_vars = {a[0]:all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1]==t)}
                                    z_vars2 = {}
                                    for i in z_vars:
                                        if i not in z_vars2:
                                            for j in model._nodes:
                                                if (i,j) in x_arcs_dict:
                                                    z_vars2[i] = z_vars[i]
                                    z_vars = z_vars2
                                found_sec = eliminate_subtours_components(model, x_arcs_dict,z_vars,t, mipsol=False)
                                if found_sec == True:
                                    ncut+=1
                                    if ncut>9:
                                        return True
           
                
                


"""
def input_ordering(df, nodes):
    def distance(city1, city2):
        c1 = coordinates[city1]
        c2 = coordinates[city2]
        diff = (c1[0]-c2[0], c1[1]-c2[1])
        return np.round(math.sqrt(diff[0]*diff[0]+diff[1]*diff[1]),0)
    df["points"] = df[["x","y"]].values.tolist()
    
    coordinates = customers_df.set_index("id")["points"].to_dict()
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
    return df"""


def input_ordering_demand(df, nodes):
    # Reset index to preserve original order as a column
    df = df.reset_index().rename(columns={'index': 'original_index'})
    
    # Sort by demand (descending), then start_inv (descending), then original index (ascending)
    df = df.sort_values(['demand', 'start_inv', 'original_index'], 
                       ascending=[False, False, True])
    
    # Assign new IDs
    df["id"] = range(1, df.shape[0] + 1)
    
    # Drop the original_index column and reset index
    df = df.drop('original_index', axis=1).reset_index(drop=True)
    
    return df

def apply_mip_start_from_construction(main_model, construction_model, christofides_routes):
    """
    Apply construction procedure solution as MIP start to main model
    
    Args:
        main_model: The main optimization model (model2_HPRP_...)
        construction_model: The solved construction procedure model
        christofides_routes: Routes generated from Christofides algorithm
        
    Returns:
        bool: True if MIP start was successfully applied
    """
    start_time = time.time()
    
    
    # Get variable mappings
    main_vars = {var.VarName: var for var in main_model.getVars()}
    construction_vars = {var.VarName: var for var in construction_model.getVars()}
    
    print(f"Setting MIP start for {len(main_vars)} variables...")
    
    # Step 1: Initialize ALL variables to zero first
    for var_name, var in main_vars.items():
        var.start = 0.0
    
    mip_start_count = 0
    
    # Step 2: Set non-routing variables from construction model
    for var_name, construction_var in construction_vars.items():
        if "z" not in var_name and "x" not in var_name:
            if var_name in main_vars and construction_var.x > 1e-6:
                main_vars[var_name].start = round(construction_var.x,0)
                mip_start_count += 1
    
    # Step 3: Create mappings for routing variables
    x_var_mapping = {}
    z_var_mapping = {}
    
    for var_name, var in main_vars.items():
        if var_name.startswith('x['):
            try:
                indices_str = var_name[2:-1]  # Remove 'x[' and ']'
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 4:
                    i, j, t, k = indices
                    x_var_mapping[(i, j, t, k)] = var
            except:
                continue
                
        elif var_name.startswith('z['):
            try:
                indices_str = var_name[2:-1]  # Remove 'z[' and ']'
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 3:
                    i, t, k = indices
                    z_var_mapping[(i, t, k)] = var
                elif len(indices) == 2:
                    i, t = indices
                    z_var_mapping[(i, t)] = var
            except:
                continue
    
    print(f"Found {len(x_var_mapping)} x variables and {len(z_var_mapping)} z variables")
    
    # Step 4: Set routing variables based on Christofides routes
    route_vars_set = 0
    used_edges = set()
    used_nodes = set()
    
    if christofides_routes:
        print("Setting route variables from Christofides solution...")
        for (period, vehicle), (route, distance) in christofides_routes.items():
            print(f"Setting route variables for Period {period}, Vehicle {vehicle}: {route}")
            
            # Set z variables for nodes in the route
            for node in route:
                if z_small==True:
                    if (node, period) in z_var_mapping:
                        z_var_mapping[(node, period)].start = 1.0
                        used_nodes.add((node, period))
                        route_vars_set += 1
                else:
                    if (node, period, vehicle) in z_var_mapping:
                        z_var_mapping[(node, period, vehicle)].start = 1.0
                        used_nodes.add((node, period, vehicle))
                        route_vars_set += 1

            
            # Set x variables for edges in the route
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i + 1]
                
                if (node_from, node_to, period, vehicle) in x_var_mapping:
                    x_var_mapping[(node_from, node_to, period, vehicle)].start = 1.0
                    used_edges.add((node_from, node_to, period, vehicle))
                    route_vars_set += 1
                    print(f"  Set x[{node_from},{node_to},{period},{vehicle}] = 1.0")
    
    # Step 5: Explicitly set all unused routing variables to zero
    unused_x_vars = 0
    unused_z_vars = 0
    
    # Set unused x variables to zero
    for key, var in x_var_mapping.items():
        if key not in used_edges:
            var.start = 0.0
            unused_x_vars += 1
    
    # Set unused z variables to zero  
    for key, var in z_var_mapping.items():
        if key not in used_nodes:
            var.start = 0.0
            unused_z_vars += 1
    
    print(f"MIP Start Summary:")
    print(f"  Non-routing variables set: {mip_start_count}")
    print(f"  Route variables set to 1: {route_vars_set}")
    print(f"  Unused x variables set to 0: {unused_x_vars}")
    print(f"  Unused z variables set to 0: {unused_z_vars}")
    print(f"  Total variables: {len(main_vars)}")
    print(f"  MIP start setup time: {time.time() - start_time:.2f} seconds")
    
    return True
def create_main_model_with_mipstart(file_path,main_model, recourse_gamma=3, stoch=True):
    """
    Create main model with MIP start from construction procedure using generate_initial_solution
    
    Args:
        file_path: Path to instance file
        recourse_gamma: Recourse cost multiplier  
        stoch: Whether to use stochastic setting
        use_construction: Whether to use construction procedure for MIP start
        construction_time_limit: Time limit for construction procedure
        
    Returns:
        tuple: (main_model, main_env, construction_success)
    """
    construction_success = False
    construction_model = None
    christofides_routes = None
    
    print(f"Running construction procedure for {file_path}")
    start_time = time.time()
    
    # Use your generate_initial_solution function
    construction_model, christofides_routes = generate_initial_solution(
        file_path, 
        recourse_gamma=recourse_gamma, 
        stoch=stoch,
        het_model=True
    )
    
    construction_time = time.time() - start_time
    
    if construction_model.SolCount > 0:
        print(f"Construction procedure found solution in {construction_time:.2f} seconds")
        print(f"Construction objective: {construction_model.ObjVal:.2f}")
        construction_success = True
    else:
        print(f"Construction procedure failed to find solution in {construction_time:.2f} seconds")
        construction_success = False
                

    
    # Apply MIP start if construction was successful
    print("Applying construction solution as MIP start...")
    mip_start_applied = apply_mip_start_from_construction(
        main_model, 
        construction_model, 
        christofides_routes
    )
    construction_time = time.time() - start_time
    
    if mip_start_applied:
        print("MIP start successfully applied to main model")
    else:
        print("Failed to apply MIP start to main model")
        construction_success = False,construction_time
    
    # Clean up construction model
    try:
        construction_model._env.close()
    except:
        pass
    
    return construction_success,construction_time



#filenames =["./Instances/Data_Test/MVPRP_C25_P6_V4_I1.dat"]

#filenames =["./Instances/Data_Test/MVPRP_C15_P9_V3_I1.dat"]
#filenames =["./Instances/Data_Test/MVPRP_C10_P6_V4_I1.dat"]

#filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C10_P9_V2_I1.dat"]

#filenames =["./Instances/Data_Test/MVPRP_C5_P2_V2_I1.dat"]

#filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P3_V4_I1.dat"]
#filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V4_I1.dat"]


def run_model2(filepath, results=[],state=None,
               recourse_gamma = 2.0,
               fixed_own = 100,
               fixed_crowd = 50,
               variable_crowd_factor = 0.5,
               r_payment = 0):
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
        
        # Every day farms requirements
        periods = [*range(1,num_days+1)]
        periodsI = [*range(0,num_days+1)]
        
            
        daily_demand = {(i,t):customers_df.set_index("id")["demand"].to_dict()[i] for i in customers for t in periods}
        
        
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
        r = {i:r_payment for i in nodes}
        r[0] = 0
        if stoch == True:
            if VARIABLE_COST==False:
                r = {}
                for i in customers:
                    r[i] = dist[0,i]*recourse_gamma
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
                #base setting Torres
                
                c = {}
                for k in vehicles:
                    for i in nodes:
                        for j in nodes:
                            if i!=j:
                                if i>0:
                                    if k>0:
                                        if det_vehicles==4 and k==1:
                                            c[i,j,k] = (distance(i,j)+r[j])*recourse_gamma*(1-sum(probabilities[l] for l in range(k+1,num_vehicles+1)))+(sum(probabilities[l] for l in range(k+1,num_vehicles+1))*(variable_crowd_factor*distance(i,j)+r[j]))
                                        else:
                                            c[i,j,k] = (distance(i,j)+r[j])*recourse_gamma*(1-sum(probabilities[l] for l in range(k,num_vehicles+1)))+(sum(probabilities[l] for l in range(k,num_vehicles+1))*(variable_crowd_factor*distance(i,j)+r[j]))
                                    else:
                                        c[i,j,k] =  (distance(i,j)+r[j])*recourse_gamma
                                else:
                                    if k>0:
                                        if det_vehicles==4 and k==1:
                                            c[i,j,k] = (fixed_own+distance(i,j)+r[j])*recourse_gamma*(1-sum(probabilities[l] for l in range(k+1,num_vehicles+1)))+(sum(probabilities[l] for l in range(k+1,num_vehicles+1))*(fixed_crowd+variable_crowd_factor*distance(i,j)+r[j]))
                                        else:
                                            c[i,j,k] =(fixed_own+distance(i,j)+r[j])*recourse_gamma*(1-sum(probabilities[l] for l in range(k,num_vehicles+1)))+(sum(probabilities[l] for l in range(k,num_vehicles+1))*(fixed_crowd+variable_crowd_factor*distance(i,j)+r[j]))
                                    else:
                                        c[i,j,k] = (fixed_own+distance(i,j)+r[j])*recourse_gamma
        else:
            if VARIABLE_COST==False:
                recourse_gamma = 0
                recourse_gamma = 0
                fixed_own = 0
                fixed_crowd = 0
                variable_cost = 0.0
                c = {(c1, c2,k): distance(c1, c2) for c1 in nodes for c2 in nodes if c1!=c2 for k in vehicles}
            else:
                #deterministic variable cost, all vehicles by own company
                c = {(c1, c2,k): distance(c1,c2)+r[c2] for c1 in nodes for c2 in nodes if c1!=c2 for k in vehicles}
                for k in vehicles:
                    for j in customers:
                        c[0,j,k] = c[0,j,k]+fixed_own
        
        
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
        if VARIABLE_COST==True:
            cost_type= "var_cost"
        else:
            cost_type = "fixed_cost"
        model_name = f'model2_HPRP_symmBreakNew3_InputOrderingDemand_USerCuts_newSymmetryBreaking_{model_type}_{cost_type}'
        if z_small == True:
            model_name += "z_reduced"
        if MIPstart == True:
            model_name += "_MIPstart"
        m = gp.Model(model_name, env=env)
        if state == None:
            state = SolverState()

        m._solver_state = state
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
        if z_small == True:
            var_z = m.addVars(customers,periods, vtype=GRB.BINARY, name='z')
        else:
            var_z = m.addVars(nodes,periods, vehicles, vtype=GRB.BINARY, name='z')
        #number of vehicles that leave the production plant in period t
        #var_k = m.addVars(periods,lb=0.0,ub=len(vehicles), vtype=GRB.CONTINUOUS, name='k')
        var_k = m.addVars(periods,vehicles, vtype=GRB.BINARY, name='k')
        #capacity flow for uncapcitated vehicle
        if VARIABLE_COST==True:
            var_f = m.addVars(dist,periods,lb=0.0, vtype=GRB.CONTINUOUS, name='F')
        
        
        
        m.modelSense = GRB.MINIMIZE
        fixed_production_costs = quicksum(setup_prod_cost*var_y[t] for t in periods)
        variable_production_costs = quicksum(var_prod_cost*var_p[t] for t in periods)
        inventory_holding_costs = quicksum(holding_costs[i]*var_I[i,t] for i in nodes for t in periods)
        routing_costs = quicksum(c[i,j,k] * var_x[i, j,t,k] for i in nodes for j in nodes if i!=j for t in periods for k in vehicles)
        m.setObjective(fixed_production_costs +variable_production_costs+inventory_holding_costs+routing_costs)
        
        m.addConstrs((var_I[0,t-1]+var_p[t] == var_I[0,t]+quicksum(var_q[i,t,k] for i in nodes for k in vehicles) for t in periods), name="Inventory balance at plant")
        m.addConstrs((var_I[i,t-1]+quicksum(var_q[i,t,k] for k in vehicles) == var_I[i,t]+daily_demand[i,t] for i in customers for t in periods), name="Inventory balance at customers")
        m.addConstrs((var_I[0,t] <= inv_cap[0] for t in periods), name="Inventory balance at plant")
        m.addConstrs((var_I[i,t-1]+quicksum(var_q[i,t,k] for k in vehicles) <= inv_cap[i] for i in customers for t in periods), name="Inventory capacity at client after delivery")
        m.addConstrs((var_p[t] <= min(production_capacity[t],sum(daily_demand[i,l] for i in customers for l in periods[t-1:]))*var_y[t] for t in periods), name="Production capacity at plant")
        
        if z_small == True:
            m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],Q,sum(daily_demand[i,l] for l in periods[t-1:]))*quicksum(var_x[i,j,t,k] for j in nodes if j!=i) for i in customers for t in periods for k in vehicles if k!=0), name="Min quantity delivered only if customer is visited in same period")
            if stoch==True:
                m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],sum(daily_demand[i,l] for l in periods[t-1:]))*quicksum(var_x[i,j,t,k] for j in nodes if j!=i) for i in customers for t in periods for k in vehicles if k==0), name="Min quantity delivered only if customer is visited in same period Uncapacitated vehicle")
            m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i for k in vehicles) == var_z[i,t] for i in customers for t in periods), name="customer visit link")
            #m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i for k in vehicles)<=1 for i in customers for t in periods), name="customer visit at most once")
            m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i)-quicksum(var_x[j,i,t,k] for j in nodes if j!=i) == 0 for i in nodes for t in periods for k in vehicles), name="degree constraints at client")
            m.addConstrs((var_x[i,j,t,k]+var_x[j,i,t,k]  <= var_z[i,t] for i in customers for j in customers if j!=i for t in periods for k in vehicles), name="subtours of size 2")
            #m.addConstrs((var_z[i,t]   <= quicksum(var_x[0,j,t,k] for j in customers for k in vehicles) for i in customers for t in periods), name="vehicle_symmetry_LeavingDepot")
            m.addConstrs((1 >= quicksum(var_x[0,j,t,k] for j in customers) for k in vehicles if k!=0 for t in periods), name="Capacity Constraints")
    
            m.addConstrs((quicksum(var_q[i,t,k] for i in customers) <= Q*quicksum(var_x[0,j,t,k] for j in customers) for k in vehicles if k!=0 for t in periods), name="Capacity Constraints")
        
            if stoch==True and  det_vehicles==2:
                
                #det_vehicles-1
                m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)   <= quicksum(var_x[0,j,t,k-1] for j in customers) for t in periods for k in vehicles[2:]), name="vehicle_symmetry1")
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,1] for t in periods for k in vehicles[2:]), name="vehicle_symmetry_vehicle3")    
        
            elif stoch==True and  det_vehicles==4:
                
                #det_vehicles-1
                m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)   <= quicksum(var_x[0,j,t,1] for j in customers) for t in periods for k in [3]), name="vehicle_symmetry3")
                m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)   <= quicksum(var_x[0,j,t,k-1] for j in customers) for t in periods for k in vehicles[3:]), name="vehicle_symmetry1")
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,1] for t in periods for k in [3]), name="vehicle_symmetry_vehicle3")
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[3:]), name="vehicle_symmetry1")
                m.addConstrs((quicksum(var_x[i,j,t,k] for j in customers if j!=i)   <= quicksum(var_x[l,j,t,k-1] for l in customers if l<i for j in customers if j!=l) for i in customers for t in periods for k in [2]), name="vehicle_symmetry2")
                #m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,kl] for l in customers if l<i for kl in [1,2]) for i in customers[1:] for t in periods for k in [det_vehicles-1]), name="vehicle_symmetry2")
                #m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k] for l in customers if l<i) for i in customers[1:] for t in periods for k in vehicles[det_vehicles:]), name="vehicle_symmetry2")
            else:
                m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)   <= quicksum(var_x[0,j,t,k-1] for j in customers) for t in periods for k in vehicles[1:]), name="vehicle_symmetry1")
            
        else:
            m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],Q,sum(daily_demand[i,l] for l in periods[t-1:]))*var_z[i,t,k] for i in customers for t in periods for k in vehicles if k!=0), name="Min quantity delivered only if customer is visited in same period")
            if stoch==True:
                m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],sum(daily_demand[i,l] for l in periods[t-1:]))*var_z[i,t,k] for i in customers for t in periods for k in vehicles if k==0), name="Min quantity delivered only if customer is visited in same period Uncapacitated vehicle")
            m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i) == var_z[i,t,k] for i in customers for t in periods for k in vehicles), name="customer visit link")
            m.addConstrs((quicksum(var_z[i,t,k] for k in vehicles)<=1 for i in customers for t in periods), name="customer visit at most once")
            if VARIABLE_COST==True:
                m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i)+quicksum(var_x[j,i,t,k] for j in nodes if j!=i) == 2*var_z[i,t,k] for i in nodes for t in periods for k in vehicles if k>0), name="degree constraints at client")
            else:
                m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i)+quicksum(var_x[j,i,t,k] for j in nodes if j!=i) == 2*var_z[i,t,k] for i in nodes for t in periods for k in vehicles), name="degree constraints at client")
            #m.addConstrs((quicksum(var_x[1,j,t,k] for j in customers)+quicksum(var_x[j,1,t,k] for j in customers) == 2*var_k[t,k] for t in periods for k in vehicles), name="degree constraints at depot")
            #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers )  quicksum(var_x[j,0,t,k] for j in customers) == 2*var_z[0,t,k] for t in periods for k in vehicles), name="leave at most once")
            #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers) == for t in periods for k in vehicles), name="degree constraints at depot")
            
            m.addConstrs((quicksum(var_q[i,t,k] for i in customers) <= Q*var_z[0,t,k] for k in vehicles if k!=0 for t in periods), name="Capacity Constraints")
            #m.addConstrs((quicksum(var_q[i,t,k] for i in customers)   <= quicksum(min(inv_cap[i],sum(daily_demand[i,l] for l in periods[t-1:])) for i in customers)*var_z[0,t,k] for i in customers for t in periods for k in vehicles if k==0), name="Capacity Constraints Vehicle")
        
            #if stoch==False:
                #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)  <= quicksum(var_x[0,j,t,k-1] for j in customers)  for t in periods for k in vehicles[1:]), name="vehicle_symmetry")
        
            #m.addConstrs((quicksum(var_x[1,j,t,k] for j in customers)== var_k[t,k] for t in periods for k in vehicles), name="degree constraints at depot 1")
            #m.addConstrs((quicksum(var_x[j,1,t,k] for j in customers)== var_k[t,k] for t in periods for k in vehicles), name="degree constraints at depot 2")
            #m.addConstrs((var_z[i,t,k] <= var_k[t,k]  for t in periods for i in customers for k in vehicles), name="Customer only visited in period t if at least one vehicle leaves the plant")
            
            m.addConstrs((var_z[i,t,k]   <= var_z[0,t,k] for i in customers for t in periods for k in vehicles), name="vehicle_symmetry_LeavingDepot")
        
            #if stoch==True and  det_vehicles==2:
                #pass
                #det_vehicles-1
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[2:]), name="vehicle_symmetry_vehicle3")    
        
            if stoch==True and  det_vehicles==4:
                
                #det_vehicles-1
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,1] for t in periods for k in [3]), name="vehicle_symmetry_vehicle3")
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[1:]), name="vehicle_symmetry1")
                m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k-1] for l in customers if l<i) for i in customers for t in periods for k in [2]), name="vehicle_symmetry2")

                #m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,kl] for l in customers if l<i for kl in [1,2]) for i in customers[1:] for t in periods for k in [det_vehicles-1]), name="vehicle_symmetry2")
                #m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k] for l in customers if l<i) for i in customers[1:] for t in periods for k in vehicles[det_vehicles:]), name="vehicle_symmetry2")
        
            else:
                if stoch==False:
                    m.addConstrs((var_z[i,t,k]   <= quicksum(var_z[l,t,k-1] for l in customers if l<i) for i in customers[1:] for t in periods for k in vehicles[1:]), name="vehicle_symmetry2")
                #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[1:]), name="vehicle_symmetry1")
        
            
            #if VARIABLE_COST==True:
            #we need subtours for zero vehicle, too!
            m.addConstrs((var_x[i,j,t,k]+var_x[j,i,t,k]  <= var_z[i,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles), name="subtours of size 2")
            #flow of homogenous vehicles
            if 0 in vehicles:
                m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i)+quicksum(var_x[j,i,t,k] for j in nodes if j!=i) == 2*var_z[i,t,k] for i in customers for t in periods for k in [0]), name="degree constraints at client")
                m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)-quicksum(var_x[j,0,t,k] for j in customers) == 0 for t in periods for k in [0]), name="degree constraints at plant")
                m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers) <= num_vehicles for t in periods for k in [0]), name="max_vehicle_link")
                m.addConstrs((quicksum(var_f[j,i,t] for j in nodes if j!=i)-quicksum(var_f[i,j,t] for j in nodes if j!=i)  == quicksum(var_q[i,t,k] for k in [0]) for i in customers for t in periods), name="Capacity Flow")
                m.addConstrs((quicksum(var_f[j,i,t] for j in nodes if j!=i)-quicksum(var_f[i,j,t] for j in nodes if j!=i)  == -quicksum(var_q[l,t,k] for l in customers for k in [0]) for i in [0] for t in periods), name="Capacity Flow Depot")
                m.addConstrs((var_f[i,j,t]  <= Q*var_x[i,j,t,k]  for i in nodes for j in nodes if i!=j for t in periods for k in [0]), name="Capacity Flow Max Depot")
            #else:
               # m.addConstrs((var_x[i,j,t,k]+var_x[j,i,t,k]  <= var_z[i,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles if k>0), name="subtours of size 2")
            
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
        """
        k_id = list()
        for e in [(0, 6), (6, 8), (8, 10), (10, 5), (5, 7), (7, 9), (9, 0)]:
            m.addConstr(var_x[e[0],e[1],2,1]==1)
            k_id.append((e[0],e[1],1,1))
        for e in [(0, 1), (1, 2), (2, 6), (6, 3), (3, 8), (8, 10), (10, 4), (4, 5), (5, 7), (7, 9), (9, 0)]:
            m.addConstr(var_x[e[0],e[1],5,1]==1)
            k_id.append((e[0],e[1],3,1))"""
        """
        for e in dist:
            for t in periods:
                for k in vehicles:
                    if (e[0],e[1],t,k) not in k_id:
                        m.addConstr(var_x[e[0],e[1],t,k]==0)"""
        """          
        edgestest = []
        for (t,k),route in (#((2,1),[(0,9),(9,0)]),
                ((3,1),[(0, 6), (6, 5), (5, 7), (7, 9), (9, 0)]),
                ((5, 1), [(0, 9), (9, 7), (7, 5), (5, 4), (4, 10), (10, 8), (8, 3), (3, 2), (2, 6), (6, 0)]),
                ((6 ,1), [(0, 6), (6, 2), (2, 1), (1, 0)])):
            for e in route:
                m.addConstr(var_x[e[0],e[1],t,k]==1)
                edgestest.append((e[0],e[1],t,k))
        for i in nodes:
            for j in nodes:
                if j!=i:
                    for t in periods:
                        for k in vehicles:
                            if (i,j,t,k) not in edgestest:
                                m.addConstr(var_x[i,j,t,k]==0)#"""
                                
        """"""""""""""""""""
        #########VIS
        """"""""""""""""""""
        if VI==True:
            m.addConstrs(var_x[0,i,t,k]<=var_z[i,t,k] for i in customers for k in vehicles for t in periods)
            m.addConstrs(quicksum(var_z[i,t,k] for k in vehicles for t in range(1,t2+1))>= round(np.ceil((sum(daily_demand[i,t] for t in range(1,t2+1))-initial_inv[i]) /(min(Q,inv_cap[i])))) for i in customers for t2 in periods)
            m.addConstrs(quicksum(var_z[i,t,k] for k in vehicles for t in range(t1,t2+1))>= round(np.ceil((sum(daily_demand[i,t] for t in range(t1,t2+1))-inv_cap[i]) /(min(Q,inv_cap[i])))) for i in customers for t1 in periods for t2 in periods if t1<t2)
            m.addConstrs(quicksum(var_z[i,t,k] for k in vehicles for t in range(t1,t2+1))>= (sum(daily_demand[i,t] for t in range(t1,t2+1))-var_I[i,t1-1]) /(min(Q,inv_cap[i])) for i in customers for t1 in periods for t2 in periods if t1<t2)
            m.addConstrs(quicksum(var_z[i,t,k] for k in vehicles for t in range(t1,t2+1))>= (sum(daily_demand[i,t] for t in range(t1,t2+1))-var_I[i,t1-1]) /sum(daily_demand[i,t] for t in range(t1,t2+1)) for i in customers for t1 in periods for t2 in periods if t1<t2)
            #m.addConstrs(quicksum(var_q[i,t,k] for k in vehicles for t in periods for k in vehicles)== sum(daily_demand[i,t] for t in periods)-initial_inv[i] for i in customers)
            #only valid if plant holding cost greater equal than customer holding cost ^^
            
            t1 = num_days
            for t in periods:
                t_temp = sum(max(0,sum(daily_demand[i,j]-initial_inv[i] for j in range(1,t+1))-initial_inv[0]) for i in customers)
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
            
        
            s = sum(max(0,sum(daily_demand[i,j]-initial_inv[i] for j in range(1,t2+1))) for i in customers)    
            
            I_s = {(i,t1):max(0,initial_inv[i]-sum(daily_demand[i,t] for t in range(1,t1+1))) for i in customers for t1 in periods}
    
            
            m.addConstr(quicksum(var_z[0,t,k] for k in vehicles for t in range(1,t2+1))>= round(np.ceil(s/Q)))
            
            m.addConstrs((var_I[i,t-s1-1] >= quicksum(daily_demand[i,t-j] for j in range(0,s1+1))*(1-quicksum(var_z[i,t-j,k] for k in vehicles for j in range(0,s1+1))) for i in customers for t in periods for s1 in range(0,t)), name="Inventory inequality")
            
            m.addConstrs(var_q[i,t,k]<= inv_cap[i]-I_s[i,t] for i in customers for t in periods for k in vehicles)
                    
                
        m._x = var_x
        m._z = var_z
        m._k = var_k
        m._q = var_q
        m._I = var_I
        m._p = var_p
        m._y = var_y
        m._c = c
        m._dist = dist
        m._med = det_vehicles
        m._r = r
        m._periods = periods
        m._vehicles = vehicles
        m._nodes = nodes
        m._customers = customers
        m._num_nodes = num_nodes
        m._cb_lastnode = 0
        
        #m.write("model.lp")
        m.params.LazyConstraints = 1
        m.params.TimeLimit = 3600
        m.params.Threads = 16
        if calc_root==True:
            m.setParam('NodeLimit', 1)
        m._cb_last_lower_bound = 0.0
        m._cb_last_obj = np.inf
    
        m.update()
        construction_runtime = 0.0
        if MIPstart==True:
            
            m.params.StartNumber = 0
            created,construction_runtime  = create_main_model_with_mipstart(file_path,m, recourse_gamma=recourse_gamma, stoch=True)
            m.params.TimeLimit = 3600-construction_runtime
            m.update()
        m.optimize(callback=cb)
        
        #if m.status==3:
           #m.computeIIS()
           #m.write("infeasible_model.ilp")
          
        def check_tour_capacity(x_arcs):
            i = 0
            j = None
            sorted_tour = []
            customers = set()
            while j!=0:
                for e in x_arcs:
                    if e[0]==i:
                        i = e[1]
                        if i!=0:
                            customers.add(i)
                        j = i
                        sorted_tour.append(e)
                        break
            return sorted_tour
        
        def parse_solution_to_routes(solution_tuples):
            """
            Parse solution tuples into separate routes by (period, vehicle)
            """
            # Group edges by (period, vehicle)
            routes_by_tk = defaultdict(list)
            
            for node1, node2, period, vehicle in solution_tuples:
                key = (period, vehicle)
                routes_by_tk[key].append((node1, node2))
            
            # Convert to ordered routes
            ordered_routes = {}
            
            for (period, vehicle), edges in routes_by_tk.items():
                # Build adjacency dictionary to find route order
                adjacency = {edge[0]: edge[1] for edge in edges}
                
                # Find starting node (try to find a node that's not a destination)
                destinations = set(edge[1] for edge in edges)
                sources = set(edge[0] for edge in edges)
                start_candidates = sources - destinations
                
                if start_candidates:
                    start_node = min(start_candidates)
                else:
                    # If it's a cycle, start with the smallest node
                    start_node = min(sources)
                
                # Build ordered path
                ordered_edges = []
                current_node = start_node
                visited_edges = set()
                
                while current_node in adjacency and (current_node, adjacency[current_node]) not in visited_edges:
                    next_node = adjacency[current_node]
                    ordered_edges.append((current_node, next_node))
                    visited_edges.add((current_node, next_node))
                    current_node = next_node
                
                # Add any remaining edges (shouldn't happen in well-formed routes)
                for edge in edges:
                    if edge not in visited_edges:
                        ordered_edges.append(edge)
                
                ordered_routes[(period, vehicle)] = ordered_edges
            
            return ordered_routes
    
        def solution_to_single_csv_with_coordinates(solution_tuples, c_matrix, dist_matrix, coordinates,
                                              model_name, instance_name, experiment_date=None, base_path="routes"):
            """
            Store all routes with coordinates in a single CSV file
            """
            
            if experiment_date is None:
                experiment_date = datetime.now().strftime("%Y%m%d")
            if VARIABLE_COST==True:
                costtype = "variable_cost"
            else:
                costtype = "fixed_cost"
            
            folder_path = os.path.join(base_path, f"{model_name}_{costtype}_I3_{experiment_date}")
            os.makedirs(folder_path, exist_ok=True)
            
            routes = parse_solution_to_routes(solution_tuples)

            
            filename = f"{instance_name}_{costtype}_rg{recourse_gamma}_f{fixed_own}_fc{fixed_crowd}_vc{variable_crowd_factor}_ri{r_payment}_sol.csv"
            filepath = os.path.join(folder_path, filename)
            
            # Collect all nodes used across all routes
            all_nodes_in_routes = set()
            
            # Write route data
            with open(filepath, 'w') as f:
                # Write header
                f.write("node1,node2,stoch_distance,distance,period,vehicle,sequence_order\n")
                
                # Write all route edges
                for (period, vehicle), route_edges in routes.items():
                    for sequence_order, (node1, node2) in enumerate(route_edges, 1):
                        edge_dist1 = c_matrix[node1, node2, vehicle] if hasattr(c_matrix, '__getitem__') else c_matrix(node1, node2, vehicle)
                        edge_dist2 = dist_matrix[node1, node2] if hasattr(dist_matrix, '__getitem__') else dist_matrix(node1, node2)
                        
                        all_nodes_in_routes.add(node1)
                        all_nodes_in_routes.add(node2)
                        
                        f.write(f"{node1},{node2},{edge_dist1},{edge_dist2},{period},{vehicle},{sequence_order}\n")
                
                # Write separator
                f.write("\n# NODE COORDINATES\n")
                f.write("node_id,x_coordinate,y_coordinate\n")
                
                # Write coordinates for all nodes used in any route
                for node_id in sorted(all_nodes_in_routes):
                    if node_id in coordinates:
                        x, y = coordinates[node_id]
                        f.write(f"{node_id},{x},{y}\n")
            
            print(f"Stored all routes in single file: {filepath}")
            print(f"  - {len(routes)} routes total")
            print(f"  - Routes: {[f't{p}_k{v}' for (p, v) in routes.keys()]}")
            
            return filepath
    
                
        error = ""
        used_vehicles = set()
        prod_quantity = set()
        cost_values = []
        costs_vars = [fixed_production_costs,variable_production_costs,inventory_holding_costs,routing_costs]
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
            solution = [e for e in var_x if var_x[e].x>0.5]
            files = solution_to_single_csv_with_coordinates(solution, c, dist, coordinates, m.ModelName,file_path.split("/")[-1].strip(".dat"))
            for t in periods:
                for k in vehicles:
                    x_arcs = [(e[0],e[1]) for e in var_x if var_x[e].x>0.5 if e[2]==t and e[3]==k]
                    q_arcs = [e for e in var_q if var_q[e].x>0.5 if e[1]==t and e[2]==k]
                    if z_small==True:
                        x_arcs_dict = {(e[0],e[1]):1.0 for e in var_x if var_x[e].x>0.5 if e[2]==t and e[3]==k}
                        z_arcs = [e[0] for e in var_z if var_z[e].x>0.5 if e[1]==t and sum(x_arcs_dict.get((e[0],j),0) for j in nodes if j!=e[0])==1]
                    else:
                        z_arcs = [e[0] for e in var_z if var_z[e].x>0.5 if e[1]==t and e[2]==k]
                    load = sum(var_q[e].x for e in q_arcs if e[0] in z_arcs)
                    if len(x_arcs)>0:
                            print(t,k, check_tour_capacity(x_arcs))
                            used_vehicles.add((t,k,load))
                    
                    if k>0:
                        if load>Q:
                            error += f"error in vehicle quantity for vehicle {k} and period {t} with load {load}>{Q}; "
                            print(f"error in vehicle quantity for vehicle {k} and period {t} with load {load}>{Q}")
                            print(q_arcs)
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
        model_runtime = m.Runtime
        if MIPstart==True:
            model_runtime += construction_runtime
        cost_labels = ["fixed_prod_cost","variable_prod_cost","inv_hold_cost","routing_cost","recourse_cost","unused_routing_cost","expected_cost_diff"]
        if VARIABLE_COST==True:
            results.append([file_path.split("/")[-1].strip(".dat"),m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime+construction_runtime,sorted(used_vehicles),Q,error,recourse_gamma,variable_crowd_factor,fixed_own,fixed_crowd,r_payment]+cost_values)
            df = pd.DataFrame(results, columns=["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recourse_gamma","variable_crowd_factor","fixed_own","fixed_crowd","r_payment"]+cost_labels)
        else:
            results.append([file_path.split("/")[-1].strip(".dat"),m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime+construction_runtime,sorted(used_vehicles),Q,error,recourse_gamma]+cost_values)
            df = pd.DataFrame(results, columns=["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recoursegamma"]+cost_labels)
        df.to_csv("results/model2_16_threads_MIPstart_I3_vss_routes_med3_gamma_fixSymmbreak.csv")
        m._solver_state.master_finished = True
        env.close()
        #model._solver_state.results = results
        #return results


def run_parallel(filepath,res):
        state = SolverState()
        #t1 = threading.Thread(target=solve_iterative,args=(barrier,env1))
        #t1 = threading.Thread(target=solve_pdptw)
        t1 = threading.Thread(target=run_model2, args=(filepath,res,state))
        t2 = threading.Thread(target=solve_ub_model, args=(filepath,state))
        t1.start()
        t2.start()
    
        t1.join()
        t2.join()
        
        return

file_path = "./Instances/Data_Test/MVPRP_C10_P3_V2_I1.dat"

file_path = "./Instances/Data_Test/MVPRP_C25_P9_V2_I1.dat"

file_path ="./Instances/Data_Test/MVPRP_C10_P3_V3_I1.dat"

file_path ="./Instances/Data_Test/MVPRP_C15_P3_V3_I1.dat"

file_path ="./Instances/Data_Test/MVPRP_C10_P6_V3_I1.dat"

files = ["./Instances/Data_Test/MVPRP_C10_P3_V3_I1.dat",
         "./Instances/Data_Test/MVPRP_C15_P3_V3_I1.dat",
         "./Instances/Data_Test/MVPRP_C10_P6_V3_I1.dat",
         "./Instances/Data_Test/MVPRP_C15_P6_V3_I1.dat",
         "./Instances/Data_Test/MVPRP_C10_P9_V3_I1.dat"]



mainFolderPath = './Instances/Data_Test/' 

#mainFolderPath = './Instances/Data_Missing/'
mainFolderPath = "./Instances/MVPRP_DatasetI3"
#mainFolderPath = "./Instances/DATA_MVPRP_Rev/"
#mainFolderPath = "./Instances/MVPRP_Dataset_Rev/"

folder = os.fsencode(mainFolderPath)
filenames = []
for subdir, dirs, files in os.walk(mainFolderPath):
    for file in files:
        if "_V3_" in file:
        #if "_V3_" in file and ("_I3" in file or "_I1" in file):
            experiment_folder = subdir.split(os.sep)[-1]
            filepath = os.path.join(subdir, file)
            filenames.append(filepath)
            
#filenames =["./Instances/MVPRP_DatasetI3/MVPRP_C40_P3_V4_I3.dat"]
#filenames =["./Instances/Data_Test/MVPRP_C15_P9_V3_I1.dat"]
#filenames =["./Instances/DATA_MVPRP_Rev/MVPRP_C10_P3_V2_I1.dat"]
#filenames = [mainFolderPath+"/MVPRP_C15_P6_V3_I3.dat"]
res = []
settings = [{"recourse_gamma" : 2.5,
    "fixed_own" : 100,
    "fixed_crowd" : 50,
    "variable_crowd_factor" : 0.5,
    "r_payment" : 0},
    {"recourse_gamma" : 1.5,
        "fixed_own": 100,
       "fixed_crowd" : 50,
        "variable_crowd_factor" : 0.5,
        "r_payment" : 0},
    {"recourse_gamma" : 2.0,
        "fixed_own": 100,
       "fixed_crowd" : 50,
        "variable_crowd_factor" : 0.25,
        "r_payment" : 0},
    {"recourse_gamma" : 2.0,
        "fixed_own": 100,
        "fixed_crowd" : 50,
        "variable_crowd_factor" : 0.75,
        "r_payment" : 0},
    {"recourse_gamma" : 2.0,
        "fixed_own": 100,
        "fixed_crowd" : 25,
        "variable_crowd_factor" : 0.5,
        "r_payment" : 0},
    {"recourse_gamma" : 2.0,
        "fixed_own": 100,
        "fixed_crowd" : 75,
        "variable_crowd_factor" : 0.5,
        "r_payment" : 0}]
settings = [{"recourse_gamma" : 3.0,
    "fixed_own" : 0,
    "fixed_crowd" : 0,
    "variable_crowd_factor" : 0.0,
    "r_payment" : 0}]

settings = [{"recourse_gamma" : 1.5,
    "fixed_own" : 0,
    "fixed_crowd" : 0,
    "variable_crowd_factor" : 0.0,
    "r_payment" : 0},
            {"recourse_gamma" : 2.0,
                "fixed_own" : 0,
                "fixed_crowd" : 0,
                "variable_crowd_factor" : 0.0,
                "r_payment" : 0},
            {"recourse_gamma" : 2.5,
                "fixed_own" : 0,
                "fixed_crowd" : 0,
                "variable_crowd_factor" : 0.0,
                "r_payment" : 0}]

for setting in settings:
    recourse_gamma = setting["recourse_gamma"]
    fixed_own = setting["fixed_own"]
    fixed_crowd = setting["fixed_crowd"]
    variable_crowd_factor = setting["variable_crowd_factor"]
    r_payment = setting["r_payment"]
    for file_path in filenames:
        if UB==True:
            run_parallel(file_path,res)
        else:
            run_model2(file_path,res,recourse_gamma=recourse_gamma,fixed_own=fixed_own,fixed_crowd=fixed_crowd,variable_crowd_factor=variable_crowd_factor,r_payment=r_payment)
    
    
    
"""
Validation
filenames =["./Instances/Data_Test/MVPRP_C10_P3_V2_I1.dat"]


cost0 = fixed_production_costs.getValue()+variable_production_costs.getValue()+inventory_holding_costs.getValue()

a1 = [(0, 1), (1, 5), (5, 2), (2, 6), (6, 4), (4, 8), (8, 9), (9, 3), (3, 0)]
S1 = set(a[0] for  a in a1  if a[0]!=0)
S2 = set([7,10,3])

cost_rec = sum(r[j] for j in S1)+sum(r[j] for j in S2)

(cost0+cost1)*0.9+0.1*(cost_rec+cost0)


"""

