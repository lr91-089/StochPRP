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
import glob
from pathlib import Path
import os
import heapq
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
#import igraph as ig
from datetime import datetime
from math import comb


import gurobipy as gp
from gurobipy import Model, GRB, quicksum, tuplelist
#from utilities_RCC_Sep_Directed_default import separate_Rounded_Capacity_cuts
from cap_sep_mip import separate_fractional_capacity_inequalities
#from RCCPSymm import separate_Rounded_Capacity_cuts_symm
#from RCCPAsymm import separate_Rounded_Capacity_cuts_symm
from construction_procedure import generate_initial_solution




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


"""
def find_capacity_subtours(model,comp, q_vars,G,capacity_cuts, period):
    G = nx.subgraph(G,comp)
    tours = []
    if len(list(G.successors(0)))>1:
        for succ in G.successors(0):
            cut_set = set()
            curr = succ
            while curr!=0:
                cut_set.add(curr)
                curr = list(G.successors(curr))[0]
            tours.append(cut_set)
    else:
        tours.append(comp)
    for tour in tours:
        if sum(q_vars[i] for i in tour if i>0)>model._Q:
            ordered_comp = tuple(sorted(tour))
            print(ordered_comp, sum(q_vars[i] for i in ordered_comp if i>0))
            if ordered_comp in capacity_cuts:
                capacity_cuts[ordered_comp] = capacity_cuts[ordered_comp].add(period)
            else:
                capacity_cuts[ordered_comp] = set((period,))
    return capacity_cuts

def eliminate_subtours_components(model, edges,q_vars, period,capacity_cuts):
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e[0],e[1],capacity=edges[e]["capacity"])
    #G.add_edges_from(list(edges.keys()),edges)
    scc = [
        c
        for c in sorted(nx.strongly_connected_components(G), key=len)
    ]
    added_st = False
    for comp in scc:
        if len(comp)>1:
            if 0 not in comp:
                if added_st==False:
                    comp_edges = list(combinations(comp,2))
                    comp_edges += [(j,i) for i,j in comp_edges]
                    subG = G.subgraph(comp)
                    #print("st lazy cut added", len(comp))
                    #glob_st_cuts.append(len(comp))
                    model.cbLazy(
                        quicksum(model._x[i, j,period] for (i,j) in comp_edges)
                        <= len(comp)-1
                    )
                    added_st = True
            else:
                if modelCut == True:
                    capacity_cuts = find_capacity_subtours(model,comp, q_vars,G,capacity_cuts, period)
    return capacity_cuts
"""

def apply_mip_start_from_construction_model1(main_model, construction_model, christofides_routes):
    """
    Apply construction procedure solution as MIP start to Model1 (homogeneous vehicle version)
    
    Args:
        main_model: The main optimization model (model1_ScenarioPRP_...)
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
        if "p" in var_name or "I" in var_name or "y" in var_name:
            if var_name in main_vars and construction_var.x > 1e-6:
                #print(var_name,construction_var.x)
                main_vars[var_name].start = round(construction_var.x,0)
                mip_start_count += 1
        elif "q" in var_name:
            if var_name[:-3]+"]" in main_vars and construction_var.x > 1e-6:
                #print(var_name[:-3]+"]",construction_var.x)
                main_vars[var_name[:-3]+"]"].start = round(construction_var.x,0)
                mip_start_count += 1
                
                
    # Step 3: Create mappings for routing variables (Model1 has no vehicle index)
    x_var_mapping = {}
    z_var_mapping = {}
    
    
    for var_name, var in main_vars.items():
        if var_name.startswith('x['):
            try:
                indices_str = var_name[2:-1]  # Remove 'x[' and ']'
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 3:  # Model1 format: x[i,j,t] (no vehicle index)
                    i, j, t = indices
                    x_var_mapping[(i, j, t)] = var
            except:
                continue
                
        elif var_name.startswith('z['):
            try:
                indices_str = var_name[2:-1]  # Remove 'z[' and ']'
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 2:  # Model1 format: z[i,t] (no vehicle index)
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
            
            # Set z variables for nodes in the route (without vehicle index for Model1)
            for node in route:
                if (node, period) in z_var_mapping:
                    z_var_mapping[(node, period)].start = 1.0
                    used_nodes.add((node, period))
                    route_vars_set += 1
            
            # Set x variables for edges in the route (without vehicle index for Model1)
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i + 1]
                
                if (node_from, node_to, period) in x_var_mapping:
                    x_var_mapping[(node_from, node_to, period)].start = 1.0
                    used_edges.add((node_from, node_to, period))
                    route_vars_set += 1
                    print(f"  Set x[{node_from},{node_to},{period}] = 1.0")
    
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

def create_main_model_with_mipstart_model1(file_path, main_model, recourse_gamma=3, stoch=True):
    """
    Create main model with MIP start from construction procedure for Model1
    
    Args:
        file_path: Path to instance file
        main_model: The main Model1 optimization model
        recourse_gamma: Recourse cost multiplier  
        stoch: Whether to use stochastic setting
        
    Returns:
        tuple: (construction_success, construction_runtime)
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
        stoch=stoch
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
    if construction_success:
        print("Applying construction solution as MIP start...")
        mip_start_applied = apply_mip_start_from_construction_model1(
            main_model, 
            construction_model, 
            christofides_routes
        )
        
        if mip_start_applied:
            print("MIP start successfully applied to main model")
        else:
            print("Failed to apply MIP start to main model")
            construction_success = False
    
    # Clean up construction model
    try:
        construction_model._env.close()
    except:
        pass
    
    return construction_success, construction_time


def eliminate_subtours_components(model, edges,q_vars,z_vars, period,capacity_cuts, mipsol=True):
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e[0],e[1],capacity=edges[e]["capacity"])
    #G.add_edges_from(list(edges.keys()),edges)
    scc = [
        c
        for c in sorted(nx.strongly_connected_components(G), key=len)
    ]
    added_st = False
    for comp in scc:
        if len(comp)>1:
            if 0 not in comp:
                if added_st==False:
                    z_comp = {key:z_vars[key] for key in comp}
                    comp_edges = list(combinations(comp,2))
                    comp_edges += [(j,i) for i,j in comp_edges]
                    #subG = G.subgraph(comp)
                    #print("st lazy cut added", len(comp))
                    #glob_st_cuts.append(len(comp))
                    e = max(z_comp, key=z_comp.get)
                    #if mipsol==True:
                    model.cbLazy(
                        quicksum(model._x[i, j,period] for (i,j) in comp_edges)
                        <= quicksum(model._z[i,period] for i in comp if i>0)-model._z[e,period]
                    )
                    model._added_cuts["lsec"].add(frozenset(comp))
                    """
                    else:
                        model.cbCut(
                            quicksum(model._x[i, j,period] for (i,j) in comp_edges)
                            <= quicksum(model._z[i,period] for i in comp if i>0)-model._z[e,period]
                        )
                        model._added_cuts["usec"].add(frozenset(comp))"""
                    added_st = True
                    return True, capacity_cuts
            """else:
                if modelCut == True:
                    for cycle in list(nx.simple_cycles(nx.subgraph(G,comp))):
                        if sum(q_vars.get(i,0) for i in cycle if i>0)>model._Q:
                            ordered_comp = tuple(sorted(cycle))
                            #comp_edges = list(combinations(ordered_comp [1:],2))
                            #if Q*sum(edges[i, j]["capacity"] for (i,j) in comp_edges if (i,j) in edges)> sum((Q*z_vars[i])-q_vars.get(i,0) for i in ordered_comp  if i>0)+pow(10,-3):
                            #print(ordered_comp, sum(q_vars[i] for i in cycle if i>0))
                            #if (ordered_comp,period) not in cut_set:
                            capacity_cuts.add((ordered_comp, period)) 
                            cut_set.add((ordered_comp,period))
                            return False, capacity_cuts"""
        """
    else:
        #min_cut = 2.0
        for i in G.nodes:
            if i>0:
                min_cut, partitions = nx.minimum_cut(G,0,i)
                if 0 in partitions[0]:
                    partition = partitions[1]
                else:
                    partition = partitions[0]
                if len(partition)>1:
                    if min_cut<z_vars[i]-pow(10,-4):
                        #argmax e
                        e = max(z_vars, key=z_vars.get)
                        model.cbCut(quicksum(model._x[i, j,period] for i in partition for j in set(nodes)-partition)
                        >= model._z[e,period])
                        return True, capacity_cuts"""
    return False, capacity_cuts


def build_igraph_graph(edges,n):
    g = ig.Graph(directed=True)
    nodes = set(u for e in edges for u in e).union(set([n]))
    node_map = {v: i for i, v in enumerate(nodes)}
    g.add_vertices(len(nodes))

    capacity = []
    edge_list = []
    
    for (u, v), cap in edges.items():
        if v==0:
            edge_list.append((node_map[u],node_map[n]))
            capacity.append(cap)
        else:
            edge_list.append((node_map[u],node_map[v]))
            capacity.append(cap)

    g.add_edges(edge_list)
    g.es['capacity'] = capacity
    return g,node_map

def exact_subtour_elemination(model,edges,z,n, period, mipsol=False):
    #return False
    added_st_cut = False
    g, node_map = build_igraph_graph(edges,n)
    rev_node_map = {idx: name for name, idx in node_map.items()}
    found_st = False
    cut_set = set()
    for u in node_map:
        if u not in [0,n]:
            result = g.st_mincut(node_map[u], node_map[n], capacity='capacity')
            cut_value = result.value
            partition = result.partition
            if cut_value < z[u]-pow(10,-3):
                #print("ST user cut",cut_value, u)
                for part in partition:
                    if node_map[n] in part:
                        S_comp = part
                    else:
                        S = part
                
                S = {rev_node_map[i] for i in S}
                #S.remove(0)
                #S.remove(n)
                z_comp = {key:z.get(key,0) for key in S}
                e = max(z_comp, key=z_comp.get)
                
                cut_set.add((len(S),tuple(S),e))
                #if mipsol==True:
                """
                else:
                    model.cbCut(
                        RHS
                        <= LHS
                    )
                    model._added_cuts["usec"].add(frozenset(S))"""
                found_st  = True
                #return True
    sorted_cuts =sorted(cut_set, key=lambda x: x[0])
    n = 0
    for size_S,S, e in sorted_cuts:
        RHS = quicksum(model._x[i, j,period] for i in S for j in S if j!=i)
        LHS = quicksum(model._z[i,period] for i in S if i>0)-model._z[e,period]
        model.cbLazy(
            RHS
            <= LHS
        )
        model._added_cuts["lsec"].add(frozenset(S))
        return found_st
        #n+=1
        #if n>10:   
    return found_st


#we need to add all capacity cuts, because q is a decision variable and can change the feasible solution to a new infeasible solution
def invoke_capacity_cuts(model,subsets):
    #if len(subsets)>1:
     #   subsets = {k: v for k, v in sorted(subsets.items(), key=lambda item: len(item[0]))}
    for comp, t in subsets:
        comp_edges = list(combinations(comp[1:],2))
        comp_edges += [(j,i) for i,j in comp_edges]
        #comp_periods = subsets[comp]
        #print("lazy capacity cut added", len(comp))
        model.cbLazy(
            model._Q*quicksum(model._x[i, j,t] for (i,j) in comp_edges)
            <= quicksum((model._Q*model._z[i,t])-model._q[i,t] for i in comp if i>0)
        )
        #only add the smallest capacity cut
        #return
        
"""
def eliminate_capacity_subtours_components(model, edges,q_vars, period):
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0],e[1],capacity=edges[e]["capacity"])
    #G.add_edges_from(list(edges.keys()),edges)
    scc = [
        c
        for c in sorted(nx.connected_components(G), key=len)
    ]
    for comp in scc:
        if len(comp)>1:
                    if sum(q_vars[i] for i in comp if i>0)>model._Q:
                        comp.remove(0)
                        comp_edges = list(combinations(comp,2))
                        comp_edges += [(j,i) for i,j in comp_edges]
                        subG = G.subgraph(comp)
                        print("lazy capacity cut added", len(comp),sum(q_vars[i] for i in comp if i>0))
                        #glob_st_cuts.append(len(comp))
                        model.cbLazy(
                            Q*quicksum(model._x[i, j,period] for (i,j) in comp_edges)
                            <= quicksum((Q*model._z[i,period])-model._q[i,period] for i in comp if i>0)
                        )
                        return True
    return False"""


def setup_mip_node_tracker(model):
    """
    Set up the MIP node tracker on a Gurobi model.
    
    Args:
        model: Gurobi model to track
        
    Returns:
        model: The same model with tracking parameters added
    """
    # Initialize tracking parameters on the model
    model._cb_node_counts = deque(maxlen=5)
    model._cb_timestamps = deque(maxlen=5)
    model._cb_last_sample_time = 0
    model._cb_last_log_time = 0
    model._cb_mipnode_stop = False
    model._cb_last_lower_bound = 0.0
    model._cb_last_obj = np.inf

    
    return model

def analyze_growth_rate(model):
    """
    Analyze the growth rate of unexplored nodes based on collected data.
    
    Args:
        model: Gurobi model with tracking data
    """
    # Get the samples
    node_counts = list(model._cb_node_counts)
    timestamps = list(model._cb_timestamps)
    
    # Calculate time differences
    time_diffs = np.diff(timestamps)
    
    # Calculate node count differences
    node_diffs = np.diff(node_counts)
    
    # Calculate growth rates (nodes per second)
    growth_rates = node_diffs / time_diffs
    
    # Only analyze if we have enough data points
    if len(growth_rates) >= 2:
        avg_growth_rate = np.mean(growth_rates)
        
        # Calculate trend over the available data (up to 100 seconds)
        # Linear regression on data points
        relative_times = np.array(timestamps) - timestamps[0]
        
        # Simple linear regression to get trend
        slope, intercept = np.polyfit(relative_times, node_counts, 1)
        
        # Determine if increasing or decreasing
        trend = "increasing" if slope > 0 else "decreasing"
        elapsed_seconds = relative_times[-1]
        if slope<=0:
            if model._cb_mipnode_stop == False:
                print(f"Stop separation of VIs: Slope: {slope:.2f}")
            model._cb_mipnode_stop = True
        else:
            if model._cb_mipnode_stop == True:
                print(f"Continue separation of VIs: Slope: {slope:.2f}")
            model._cb_mipnode_stop = False
        
        """
        print(f"Unexplored nodes: {node_counts[-1]}")
        print(f"Growth rate: {avg_growth_rate:.2f} nodes/second")
        print(f"Trend over last {elapsed_seconds:.1f} seconds: {trend}")
        print(f"Slope: {slope:.2f}")
        print("-" * 40)"""

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
    if where == GRB.Callback.MIP:
        # Get current time
        current_time = time.time()
        
        # Check if it's time for a new sample (every 10 seconds)
        if current_time - model._cb_last_sample_time >= 5:
            # Get the count of unexplored nodes
            unexplored_nodes  = model.cbGet(GRB.Callback.MIP_NODLFT)
            
            # Store the data
            model._cb_node_counts.append(unexplored_nodes)
            model._cb_timestamps.append(current_time)
            model._cb_last_sample_time = current_time
            
            # If we have enough data, calculate and report growth rate
            if len(model._cb_node_counts) >= 2:
                analyze_growth_rate(model)
        if current_time - model._cb_last_log_time >=20:
            model._cb_last_log_time = current_time
            log_str = ""
            for key in model._added_cuts:
                log_str += f"{key}:{len(model._added_cuts[key])} \t"
            print(log_str)
    ###MIPSOL callback
    #Q*sum(x_arcs[i, j,t] for (i,j) in list(combinations(comp[1:],2)) for t in {3})<= sum((Q*all_z_vars[i,t])-q_vars[i,t] for i in comp if i>0 for t in {3})
    if where == GRB.Callback.MIPSOL:
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        UB = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        LB = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        curr_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        #if curr_obj==22434.1:
         #   print("check cb")
        gap = (UB-LB)/UB
        all_x_vars = model.cbGetSolution(model._x)
        all_z_vars = model.cbGetSolution(model._z)
        all_q_vars = model.cbGetSolution(model._q)
        epsilon = pow(10, -4)
        x_arcs = {a:all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
        #subtours
        capacity_cuts = set()
        for t in model._periods:
            x_arcs_dict = {(a[0],a[1]):{"capacity":x_arcs[a]} for a in x_arcs if a[2]==t}
            q_vars = {a[0]:all_q_vars[a] for a in all_q_vars if (all_q_vars[a] > epsilon and a[1]==t)}
            z_vars = {a[0]:all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1]==t)}
            found_sec, capacity_cuts = eliminate_subtours_components(model, x_arcs_dict,q_vars, z_vars,t,capacity_cuts)
            if found_sec==False:
                found_fcc = separate_fractional_capacity_inequalities(model,x_arcs_dict,z_vars,q_vars,model._Q,model._n,t, mipsol=True)
                if found_fcc==True:
                    return True
            else:
                return True
                #invoke_capacity_cuts(model,capacity_cuts)
                #if len(capacity_cuts)>0 and nodecnt<100:
                    #print(capacity_cuts)
                    #separate_fractional_capacity_inequalities(model,x_arcs_dict,z_vars,q_vars,Q,num_nodes,t, mipsol=True)
    
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            if model._cb_mipnode_stop ==False:
                nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                cur_lb =  model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                cur_ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                all_x_vars = model.cbGetNodeRel(model._x)
                all_z_vars = model.cbGetNodeRel(model._z)
                all_q_vars = model.cbGetNodeRel(model._q)
                epsilon = pow(10, -4)
                x_arcs = {a:all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
                capacity_cuts = set()
                if ((nodecnt<1 and len(model._added_cuts["ufcc"])+len(model._added_cuts["usec"])<250) or nodecnt<200 or (nodecnt>200 and nodecnt - model._cb_lastnode >= 400)):
                    sep = True
                    if (nodecnt>200):
                        if model._cb_last_lower_bound<cur_lb-epsilon or model._cb_last_ub-epsilon>cur_ub:
                            sep = False
                    model._cb_last_lower_bound = cur_lb
                    model._cb_last_ub = cur_ub
                    model._cb_lastnode= nodecnt
                    if sep==True:
                        ncut = 0
                        for t in model._periods:
                            x_arcs_dict = {(a[0],a[1]):{"capacity":x_arcs[a]} for a in x_arcs if a[2]==t}
                            q_vars = {a[0]:all_q_vars[a] for a in all_q_vars if (all_z_vars[a] > epsilon and a[1]==t)}
                            z_vars = {a[0]:all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1]==t)}
                            #if nodecnt<1:
                            #found_sec = exact_subtour_elemination(model,x_arcs_dict,z_vars,model._n, t, mipsol=False)
                            found_sec, capacity_cuts = eliminate_subtours_components(model, x_arcs_dict,q_vars, z_vars,t,capacity_cuts, mipsol=False)
                            #else:
                                #found_sec, capacity_cuts = eliminate_subtours_components(model, x_arcs_dict,q_vars, z_vars,t,capacity_cuts, mipsol=False)
                            if found_sec==False:#and len(model._added_cuts["ufcc"])+len(model._added_cuts["lfcc"])<200:
                                    #print("sep fractional cuts")
                                    found_fcc = separate_fractional_capacity_inequalities(model,x_arcs_dict,z_vars,q_vars,model._Q,model._n,t)
                                    if found_fcc ==True:
                                        ncut+=1
                            else:
                                ncut +=1
                            if ncut>9:
                                return True
                        #invoke_capacity_cuts(model,capacity_cuts)

                                


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

def load_folder_data_routes(folder_path, pattern="*.csv", folder_name=""):
    """Load all CSV files from a folder, properly separating route data from coordinates"""
    files = glob.glob(os.path.join(folder_path, pattern))
    
    if not files:
        print(f"No files found in {folder_path}")
        return {}
    
    
        
    
    folder_data = {}
    print(f"Loading {len(files)} files from {folder_name}...")
    
    for file_path in files:
        instance_name = Path(file_path).stem
        
        try:
            # Read the raw file content first
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Split content at coordinate section
            lines = content.split('\n')
            route_lines = []
            coordinate_section_started = False
            
            for line in lines:
                line = line.strip()
                
                # Check if we've reached the coordinate section
                if ('# NODE COORDINATES' in line or 
                    'node_id,x_coordinate,y_coordinate' in line or
                    (coordinate_section_started and 'node_id' in line and 'coordinate' in line)):
                    coordinate_section_started = True
                    continue
                
                # If we haven't reached coordinates yet, it's route data
                if not coordinate_section_started and line:
                    route_lines.append(line)
            
            # Create DataFrame from route data only
            if route_lines:
                route_lines[0] = 'node1,node2,stoch_distance,distance,period,vehicle_num,vehicle,sequence_order'
                # Write route data to temporary string
                route_content = '\n'.join(route_lines)
                
                # Read with pandas
                from io import StringIO
                df = pd.read_csv(StringIO(route_content), dtype={'node1': str, 'node2': str})
                # Basic cleaning
                df = df.dropna(how='all')  # Remove completely empty rows
                
                # Check for required columns
                required_cols = ['node1', 'node2', 'period', 'vehicle']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"  ✗ {instance_name}: Missing columns {missing_cols}")
                    continue
                
                # Remove rows with missing essential data
                df = df.dropna(subset=['node1', 'node2'])
                
                # Additional cleaning: remove any rows that still contain coordinate-like data
                df = df[~df['node1'].astype(str).str.contains('node|coordinate', case=False, na=False)]
                df = df[~df['node2'].astype(str).str.contains('node|coordinate', case=False, na=False)]
                
                if len(df) == 0:
                    print(f"  ✗ {instance_name}: No valid route data after cleaning")
                    continue
                
                folder_data[instance_name] = df
                print(f"  ✓ {instance_name}: {len(df)} route segments")
            else:
                print(f"  ✗ {instance_name}: No route data found")
            
        except Exception as e:
            print(f"  ✗ {instance_name}: {str(e)}")
    
    print(f"Successfully loaded {len(folder_data)} instances")
    return folder_data

def analyze_routes(df, distance_col='distance'):
    """Analyze route structure in a DataFrame"""
    
    # Get unique periods and vehicles
    periods = sorted([x for x in df['period'].unique() if pd.notna(x)])
    vehicles = sorted([x for x in df['vehicle'].unique() if pd.notna(x)])
    
    routes = {}
    
    for period in periods:
        for vehicle in vehicles:
            # Filter data for this period/vehicle combination
            route_data = df[(df['period'] == period) & (df['vehicle'] == vehicle)]
            
            if len(route_data) == 0:
                continue
            
            # Sort by sequence if available
            if 'sequence_order' in route_data.columns:
                route_data = route_data.sort_values('sequence_order')
            vehicle_num = int(route_data["vehicle_num"].iloc[0])
            # Build route sequence
            route_sequence = []
            for _, row in route_data.iterrows():
                if not route_sequence:  # First node
                    route_sequence.append(str(row['node1']))
                route_sequence.append(str(row['node2']))
            
            # Calculate distance
            total_distance = 0
            if distance_col in route_data.columns:
                distances = pd.to_numeric(route_data[distance_col], errors='coerce').fillna(0)
                total_distance = distances.sum()
            
            route_key = (int(period),vehicle_num,int(vehicle))
            routes[route_key] = {
                'sequence': route_sequence,
                'distance': total_distance,
                'segments': len(route_data),
                'period': int(period),
                'vehicle': int(vehicle),
                'vehicle_num': vehicle_num
            }
    
    return routes

def fix_routes_in_master(model,fixed_routes):
    counter = 0
    fixed_arcs = []
    for t,k,kid in fixed_routes:
        route = fixed_routes[(t,k,kid)]
        print(f"Fix route in period {t} with sequence:{route['sequence']}")
        i = int(route["sequence"][0])
        for j in route["sequence"][1:]:
            j = int(j)
            model.addConstr(model._x[i,j,t,k]==1.0)
            if j in model._customers:
                model.addConstr(model._z[j,t,k]==1.0)
            fixed_arcs.append((i,j,t,k))
            i = j
        counter +=1
    #"""
    for i in model._nodes:
        for j in model._nodes:
            if j!=i:
                for t in model._periods:
                    if (i,j,t,k) not in fixed_arcs:
                        model.addConstr(model._x[i,j,t,k]==0)#"""
    print(f"Fixed {counter} routes!")
    return model

def binomial_distribution(p, k, M=100):
    """
    calculates discrete binomial distribution, 
    k is number of expected vehicles, M is the 
    pool and p is the probabiltiy of success
    """
    return comb(M, k) * (p**k) * ((1 - p)**(M - k))

def binomial_cdf(p, k, M=100):
    return sum(comb(M, i) * (p**i) * ((1 - p)**(M - i)) for i in range(k + 1))
        
    
    
def run_model(file_path,settings, results=[]):
    
    recourse_gamma = settings["recourse_gamma"]
    fixed_own = settings["fixed_own"]
    fixed_crowd = settings["fixed_crowd"]
    variable_crowd_factor = settings["variable_crowd_factor"]
    r_payment = settings["r_payment"]
    stoch = settings["stoch"]


    modelCut = settings["modelCut"]

    integral = settings["integral"]

    MIPstart = settings["MIPstart"]

    calc_root = settings["calc_root"]

    #0: no var cost, 1:fixed_vehicle in determ case or stochcase, 2:crowd_vehicle in determ case
    VARIABLE_COST = settings["VARIABLE_COST"]

    STOCH_VALUE = settings["STOCH_VALUE"]
    route_data = settings["route_data"]
    BINOM = settings["binom_dist"]
    binom_p = settings["binom_p"]
    print(file_path)

    dataFile = parse_mvprp_instance(file_path)
    
    cut_set = set()
    
    num_nodes,num_vehicles, num_days, Q = dataFile["num_nodes"],dataFile["num_vehicles"],dataFile["periods"],dataFile["Q"]
    
    depot = dataFile["supplier"]
    
    # Read remaining rows as customers
    customers = dataFile["retailers"]
        
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
    
    
    # List of nodes including the depot node 1
    
    customers = [*range(1,num_nodes)]
    
    # Every day farms requirements
    periods = [*range(1,num_days+1)]
    periodsI = [*range(0,num_days+1)]
    
        
    daily_demand = {(i,t):customers_df.set_index("id")["demand"].to_dict()[i] for i in customers for t in periods}
    
    
    if stoch == True:
        if num_vehicles == 2:
            vehicles = [num_vehicles-1,num_vehicles,num_vehicles+1,num_vehicles+2]
            probabilities = {num_vehicles-2:0.1,num_vehicles-1:0.2,num_vehicles:0.4,num_vehicles+1:0.2,num_vehicles+2:0.1}
        else:
            vehicles = [num_vehicles-2,num_vehicles-1,num_vehicles,num_vehicles+1,num_vehicles+2]
            probabilities = {num_vehicles-2:0.1,num_vehicles-1:0.2,num_vehicles:0.4,num_vehicles+1:0.2,num_vehicles+2:0.1}
        
        det_vehicles = num_vehicles
        num_vehicles = 4
        if STOCH_VALUE==True:
            num_vehicles=5
        if BINOM:
            probabilities = {}
            vehicles = []
            for k in range(num_vehicles+1):
                probabilities[k] =binomial_distribution(binom_p,k)
                vehicles.append(k)
    else:
        num_vehicles = num_vehicles
        vehicles = [*range(1,num_vehicles+1)]
        if BINOM:
            det_vehicles = num_vehicles
            num_vehicles = 5
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
    vehicle_types = []
    
    #recourse costs
    #recourse costs
    
    r = {i:r_payment for i in nodes}
    r[0] = 0
    #recourse_gamma = 3
    if stoch==True and VARIABLE_COST == 0:
        r = {}
        for i in customers:
            r[i] = dist[0,i]*recourse_gamma
    if VARIABLE_COST == 0:
        #recourse cost
        fixed_own = 0
        fixed_crowd = 0
        variable_crowd_factor = 0.0
        r_payment = 0
        if stoch==False:
            recourse_gamma = 0
            r = {i:r_payment for i in nodes}
            r[0] = 0
    
    c = dist
    vehicle_type_num = {0:num_vehicles, 1:4}
    if STOCH_VALUE==True:
        vehicle_type_num[1] = 5
    if stoch==False:
        vehicle_type_num = {0:det_vehicles, 1:5}
    if BINOM:
        vehicle_types = [0,1]
        c = {}
        for k in vehicle_types:
            for i,j in dist:
                if k==1:
                    if i>0:
                        c[i,j,k] = dist[i,j]*variable_crowd_factor
                    else:
                        c[i,j,k] = dist[i,j]*variable_crowd_factor+fixed_crowd
                else:
                    if i>0:
                        c[i,j,k] = dist[i,j]
                    else:
                        c[i,j,k] = dist[i,j]+fixed_own
    
    Ax = tuplelist([(i,j,t,k) for (i,j) in dist for t in periods for k in vehicles])
    
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
    model_name = 'model1_ScenarioPRP_homogeneousFractionalCapCut'
    if modelCut==True:
        model_name += "_capCuts"
    else:
        model_name += "_oneCommodityFlow"
    if stoch == True:
        model_name += "_stoch"
    else:
        model_name += "_determ_medPlus2"
    if integral==True:
        model_name += "_integral"
    if MIPstart==True:
        model_name += "_MIPstart"
    m = gp.Model(model_name, env=env)
    Qk = {0:Q,1:round(Q/2,0)}
    if VARIABLE_COST==1:
        vehicle_types = [0]
    # Decision variables: 
    
    # Edge variables = 1, if customer i is visited after h in period t by vehicke k
    var_x = m.addVars(dist, periods,vehicle_types, vtype=GRB.BINARY, name='x')
    #binary variable = 1 if setup occurs in period t
    var_y = m.addVars(periods, vtype=GRB.BINARY, name='y')
    #continous capacity flow
    if modelCut == False:
        var_f = m.addVars(dist,periods,vehicle_types,lb=0.0, vtype=GRB.CONTINUOUS, name='f')
    #quantity produced in period t
    var_p = m.addVars(periods,lb=0.0, vtype=GRB.CONTINUOUS, name='p')
    #quantity of inventory at node i in period t
    var_I = m.addVars(nodes,periodsI,lb=0.0, vtype=GRB.CONTINUOUS, name='I')
    #quantity delivered to cust i in period t
    var_q = m.addVars(nodes,periods,vehicle_types,lb=0.0, vtype=GRB.CONTINUOUS, name='q')
    #binary variable equal to 1 if node i in N is visited in period t
    var_z = m.addVars(nodes,periods,vehicle_types, vtype=GRB.BINARY, name='z')
    #number of vehicles that leave the production plant in period t
    #var_k = m.addVars(periods,lb=0.0,ub=len(vehicles), vtype=GRB.CONTINUOUS, name='k')
    #var_k = m.addVars(periods, lb=0.0, ub= num_vehicles, vtype=GRB.INTEGER, name='k')
    #continous variable alpha
    if stoch==True:
        if integral==False:
            var_alpha = m.addVars(dist, periods,probabilities, vtype=GRB.BINARY, name='alph')
            #contunous variable beta
            var_beta = m.addVars(customers, periods,probabilities, vtype=GRB.BINARY, name='beta')
        else:
            var_alpha = m.addVars(dist, periods,probabilities,lb=0.0,ub=1.0, vtype=GRB.CONTINUOUS, name='alph')
            #contunous variable beta
            var_beta = m.addVars(customers, periods,probabilities,lb=0.0,ub=1.0, vtype=GRB.CONTINUOUS, name='beta')

    
   
    
    
    m.modelSense = GRB.MINIMIZE
    fixed_production_costs = quicksum(setup_prod_cost*var_y[t] for t in periods)
    variable_production_costs = quicksum(var_prod_cost*var_p[t] for t in periods)
    inventory_holding_costs = quicksum(holding_costs[i]*var_I[i,t] for i in nodes for t in periods)
    if VARIABLE_COST==1:
            routing_costs = quicksum(r[i]*var_z[i,t,k] for k in vehicle_types for i in customers for t in periods)+quicksum(c[i,j,k]*var_x[i, j,t,k] for k in vehicle_types for i in nodes for j in nodes if i!=j for t in periods) 
    elif VARIABLE_COST==2:
            routing_costs = quicksum(r[i]*var_z[i,t,k] for k in vehicle_types for i in customers for t in periods)+quicksum(c[i,j,k]*var_x[i, j,t,k] for k in vehicle_types for i in nodes for j in nodes if i!=j for t in periods) 
    else:
        routing_costs = quicksum(c[i,j] * var_x[i, j,t] for i in nodes for j in nodes if i!=j for t in periods)
    if stoch==True:
        #recourse_costs = quicksum(probabilities[s]*r[i]*(var_z[i,t]-var_beta[i,t,s]) for i in customers for t in periods for s in probabilities)
        #unused_routing_costs = quicksum(probabilities[s]*c[i,j] * (var_x[i, j,t] -var_alpha[i,j,t,s]) for i in nodes for j in nodes if i!=j for t in periods for s in probabilities)
        #expected_costs = (recourse_costs-unused_routing_costs)
        if VARIABLE_COST>0:
            recourse_costs = quicksum(r[i]*probabilities[s]*(var_z[i,t,1]-var_beta[i,t,s]) for i in customers for t in periods for s in probabilities)+quicksum(probabilities[s]*c[i,j,0] * (var_x[i, j,t,1] -var_alpha[i,j,t,s]) for i in nodes for j in nodes if i!=j for t in periods for s in probabilities)
            unused_routing_costs = quicksum(r[i]*probabilities[s]*(var_z[i,t,1]-var_beta[i,t,s]) for i in customers for t in periods for s in probabilities)+quicksum(probabilities[s]*c[i,j,1] *(var_x[i, j,t,1] -var_alpha[i,j,t,s]) for i in nodes for j in nodes if i!=j for t in periods for s in probabilities)
            expected_costs = ((recourse_gamma*recourse_costs)-unused_routing_costs)
            routing_costs = quicksum(r[i]*var_z[i,t,k] for k in vehicle_types for i in customers for t in periods)+quicksum(c[i,j,k]* var_x[i, j,t,k] for k in vehicle_types for i in nodes for j in nodes if i!=j for t in periods) 


    else:
        expected_costs = 0
        recourse_costs = 0
        unused_routing_costs = 0
    m.setObjective(fixed_production_costs +variable_production_costs+inventory_holding_costs+routing_costs+expected_costs)
    
    m.addConstrs((var_I[0,t-1]+var_p[t] == var_I[0,t]+quicksum(var_q[i,t,k] for k in vehicle_types for i in nodes) for t in periods), name="Inventory balance at plant")
    m.addConstrs((var_I[i,t-1]+quicksum(var_q[i,t,k] for k in vehicle_types) == var_I[i,t]+daily_demand[i,t] for i in customers for t in periods), name="Inventory balance at customers")
    m.addConstrs((var_I[0,t] <= inv_cap[0] for t in periods), name="Inventory balance at plant")
    m.addConstrs((var_I[i,t-1]+quicksum(var_q[i,t,k] for k in vehicle_types)  <= inv_cap[i] for i in customers for t in periods), name="Inventory capacity at client after delivery")
    m.addConstrs((var_p[t] <= min(production_capacity[t],sum(daily_demand[i,l] for i in customers for l in periods[t-1:]))*var_y[t] for t in periods), name="Production capacity at plant")
    m.addConstrs((var_q[i,t,k]   <= min(inv_cap[i],sum(daily_demand[i,l] for l in periods[t-1:]))*var_z[i,t,k] for k in vehicle_types for i in customers for t in periods), name="Min quantity delivered only if customer is visited in same period")
    m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i) == var_z[i,t,k] for i in customers for k in vehicle_types for t in periods), name="customer visit link")
    m.addConstrs((quicksum(var_x[i,j,t,k] for j in nodes if j!=i)+quicksum(var_x[j,i,t,k] for j in nodes if j!=i) == 2*var_z[i,t,k] for k in vehicle_types for i in customers for t in periods), name="degree constraints at client")
    
    m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers) == quicksum(var_x[j,0,t,k] for j in customers) for t in periods for k in vehicle_types), name="degree constraints at plant")

    if stoch==False:
        m.addConstrs(quicksum(var_x[0,j,t,0] for j in customers)<= vehicle_type_num[0]*quicksum(var_x[0,j,t,1] for j in customers) for t in periods)
    m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers) <= vehicle_type_num[k] for i in customers for t in periods for k in vehicle_types), name="vehicle limit")
    
    m.addConstrs((Q*quicksum(var_x[0,j,t,k] for j in customers) >= quicksum(var_q[i,t,k] for i in customers) for t in periods for k in vehicle_types), name="min capacity inequality")


    #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[1:]), name="vehicle_symmetry1")
    
    #m.addConstrs((quicksum(var_z[i,t,k] for k in vehicles if k>i)   == 0 for t in periods for i in customers), name="vehicle_symmetryOrderCustomer")

    

    #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)  <= quicksum(var_x[0,j,t,k-1] for j in customers)  for t in periods for k in vehicles[1:]), name="vehicle_symmetry")



    
    m.addConstrs((var_x[i,j,t,k]+var_x[j,i,t,k]  <= var_z[i,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicle_types), name="subtours of size 2")
    #m.addConstrs((quicksum(i*var_x[0,i,t] for i in customers)  <= quicksum(i*var_x[i,0,t] for i in customers)  for t in periods), name="symmetric_cost_vi")

    
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
    if stoch==True:
        m.addConstrs((quicksum(var_alpha[0,j,t,s] for j in customers) <= s for s in probabilities for t in periods), name="1.19")
        
        m.addConstrs((var_alpha[i,j,t,s]  <= var_x[i,j,t,1] for (i,j) in dist for t in periods for s in probabilities), name="1.20")
        
        m.addConstrs((var_beta[i,t,s]  <= var_z[i,t,1] for i in customers for t in periods for s in probabilities), name="1.21")
        
        m.addConstrs((var_beta[i,t,s]  <= quicksum(var_alpha[i,j,t,s] for j in nodes if j!=i) for i in customers for t in periods for s in probabilities), name="1.22")
    
        m.addConstrs((quicksum(var_alpha[i,j,t,s] for j in nodes if j!=i)== quicksum(var_alpha[j,i,t,s] for j in nodes if j!=i)  for i in customers for t in periods for s in probabilities), name="1.23")

    if modelCut == False:
        m.addConstrs((quicksum(var_f[j,i,t,k] for j in nodes if j!=i)-quicksum(var_f[i,j,t,k] for j in nodes if j!=i)  ==var_q[i,t,k] for k in vehicle_types for i in customers for t in periods), name="Capacity Flow")
        
        
        m.addConstrs((quicksum(var_f[j,i,t,k] for j in nodes if j!=i)-quicksum(var_f[i,j,t,k] for j in nodes if j!=i)  == -quicksum(var_q[l,t,k] for l in customers) for i in [0] for t in periods for k in vehicle_types), name="Capacity Flow Depot")

    
    
        m.addConstrs((var_f[i,j,t,k]  <= Qk[k]*var_x[i,j,t,k]  for i in nodes for j in nodes if i!=j for t in periods for k in vehicle_types), name="Capacity Flow Max Depot")
    

    m._x = var_x
    m._z = var_z
    #m._k = var_k
    m._q = var_q
    m._Ax = Ax
    m._Q = Q
    m._vehicles = vehicles
    m._n = num_nodes
    m._nodes = nodes
    m._customers = customers
    m._periods = periods
    if STOCH_VALUE==True: 
        route_name = file_path.split("/")[-1].strip(".dat")+"_rg2.0_f100_fc50_vc0.5_ri0_CDTest"+"_sol"
        route_sequences = analyze_routes(route_data[route_name])
        m = fix_routes_in_master(m, route_sequences)
    
    #m.write("model.lp")
    m.params.TimeLimit = 3600
    m.params.Threads = 16
    m._added_cuts = {"lfcc":set(),"ufcc":set(),"lsec":set(),"usec":set()}
    m._cb_lastnode = 0
    m = setup_mip_node_tracker(m)
    if calc_root==True:
        m.setParam('NodeLimit', 1)
    # Apply MIP start if enabled
    m.update()
    construction_runtime = 0.0
    if MIPstart==True:
        m.params.StartNumber = 0
        #m.params.MIPFocus = 3
        construction_success, construction_runtime = create_main_model_with_mipstart_model1(
            file_path, m, recourse_gamma=3, stoch=stoch
        )
        m.params.TimeLimit = 3600-construction_runtime  # Reserve 10 seconds for construction
        if construction_success:
            print("Construction procedure MIP start applied successfully!")
        else:
            print("Construction procedure failed, proceeding without MIP start")
    if modelCut==True:
        m.params.LazyConstraints = 1
        #m.params.PreCrush = 1
        m.update()
        m.optimize(callback=cb)
    else:
        m.update()
        m.optimize()
    
    #if m.status==3:
     #  m.computeIIS()
      # m.write("infeasible_model.ilp")
      
    def check_tour_capacity(x_arcs,q_arcs, period, error):
        tours = []
        for arcs in x_arcs:
            if arcs[0]==0:   
                i = 0
                j = arcs[1]
                sorted_tour = [(i,j)]
                customers = set((j,))
                while j!=0:
                    for e in x_arcs:
                        if e[0]==j:
                            i = e[0]
                            if j!=0:
                                customers.add(j)
                            j = e[1]
                            sorted_tour.append(e)
                            break
                #print(customers)
                if sum(q_arcs[i] for i in customers)>Q+pow(10,-4):
                    print(f"error!, load {sum(q_arcs[i] for i in customers)} of a vehicle is over {Q}, custommers: {customers}")
                    error += f"error!, load {sum(q_arcs[i] for i in customers)} of a vehicle is over {Q}, custommers: {customers};"
                tours.append(sorted_tour)
                print(period, sorted_tour)
        return tours, error
    
    def parse_solution_to_routes_no_vehicle_with_artificial_vehicle(solution_tuples):
        """
        Parse solution tuples into separate routes by period using tour detection logic
        """
        # Group edges by period
        routes_by_t = defaultdict(list)
        
        for node1, node2, period, vehicle in solution_tuples:
            routes_by_t[period,vehicle].append((node1, node2))
        
        ordered_routes = {}
        
        for (period,k), x_arcs in routes_by_t.items():
            tours = []
            vehicle_id = 1
            
            # Find all tours starting from node 0
            processed_arcs = set()
            
            for arcs in x_arcs:
                if arcs[0] == 0 and arcs not in processed_arcs:   
                    i = 0
                    j = arcs[1]
                    sorted_tour = [(i, j)]
                    processed_arcs.add((i, j))
                    
                    while j != 0:
                        for e in x_arcs:
                            if e[0] == j and e not in processed_arcs:
                                i = e[0]
                                j = e[1]
                                sorted_tour.append(e)
                                processed_arcs.add(e)
                                break
                    
                    tours.append(sorted_tour)
                    print(period,k, sorted_tour)
                    
                    # Store with increasing vehicle index
                    ordered_routes[(period, k, vehicle_id)] = sorted_tour
                    vehicle_id += 1
        
        return ordered_routes

    def solution_to_single_csv_no_vehicle_artificial_vehicle(solution_tuples, c_matrix, dist_matrix, coordinates,
                                                           model_name, instance_name, experiment_date=None, base_path="routes"):
        """
        Store all routes with coordinates in a single CSV file (no vehicle index, but add artificial vehicle)
        """
        
        if experiment_date is None:
            experiment_date = datetime.now().strftime("%Y%m%d")
        
        folder_path = os.path.join(base_path, f"{model_name}_CD_MED3_VCD_{experiment_date}")
        os.makedirs(folder_path, exist_ok=True)
        
        routes = parse_solution_to_routes_no_vehicle_with_artificial_vehicle(solution_tuples)
        
        filename = f"{instance_name}_rg{recourse_gamma}_f{fixed_own}_fc{fixed_crowd}_vc{variable_crowd_factor}_ri{r_payment}_CDTest_sol.csv"
        filepath = os.path.join(folder_path, filename)
        
        # Collect all nodes used across all routes
        all_nodes_in_routes = set()
        
        # Write route data
        with open(filepath, 'w') as f:
            # Write header (include artificial vehicle column)
            f.write("node1,node2,stoch_distance,distance,period,vehicle,sequence_order\n")
            
            # Write all route edges
            for (period, k, vehicle), route_edges in routes.items():
                for sequence_order, (node1, node2) in enumerate(route_edges, 1):
                    # No vehicle index for c_matrix since original problem has no vehicle
                    edge_dist1 = c_matrix[node1, node2,k] if hasattr(c_matrix, '__getitem__') else c_matrix(node1, node2,k)
                    edge_dist2 = dist_matrix[node1, node2] if hasattr(dist_matrix, '__getitem__') else dist_matrix(node1, node2)
                    
                    all_nodes_in_routes.add(node1)
                    all_nodes_in_routes.add(node2)
                    
                    f.write(f"{node1},{node2},{edge_dist1},{edge_dist2},{period},{k},{vehicle},{sequence_order}\n")
            
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
        print(f"  - Routes: {[f't{p}_k{k}_kid{v}' for (p, k, v) in routes.keys()]}")
        
        return filepath
            
    error = ""
    used_vehicles = set()
    prod_quantity = set()
    cost_values = []
    costs_vars = [fixed_production_costs,variable_production_costs,inventory_holding_costs,routing_costs,recourse_costs,unused_routing_costs,expected_costs]
    if m.status==2 or m.SolCount>0:
        for var in costs_vars:
            if isinstance(var, gp.LinExpr):
                print(var.getValue())
                cost_values.append(var.getValue())
            else:
                print(var)
                cost_values.append(var)
        
        for iter2 in range(1):
                m.setParam(GRB.Param.SolutionNumber, iter2)
                print('%g ' % m.PoolObjVal, end='\n')
                for v in m.getVars():
                     if v.xn > 1e-5:
                           #print ('%s %g' % (v.varName, v.xn))
                           print ('%s %g' % (v.varName, v.xn))
                print("\n")
        print("\n")
        solution = [e for e in var_x if (var_x[e].x>0.5)]
        if STOCH_VALUE==False:
            single_file = solution_to_single_csv_no_vehicle_artificial_vehicle(
                solution, c, dist, coordinates, 
                m.ModelName, file_path.split("/")[-1].strip(".dat")
            )
        for t in periods:
            x_arcs = [(e[0],e[1]) for e in var_x if (var_x[e].x>0.5 and e[2]==t)]
            x_a2 = [e for e in var_x if (var_x[e].x>0.5 and e[2]==t)]
            q_vals = {e[0]:var_q[e].x for e in var_q if e[1]==t}
            z_arcs = [e[0] for e in var_z if  e[1]==t]
            """
            #if len(x_arcs)>0:
                #print(x_a2)
                #tours, error = check_tour_capacity(x_arcs,q_vals,t, error)
                
                used_vehicles.add((t,load))
                if load>Q:
                    error += f"error in vehicle quantity for vehicle {k} and period {t} with load {load}>{Q}; "
                    print(f"error in vehicle quantity for vehicle {k} and period {t} with load {load}>{Q}")
                    print(q_arcs)"""
    else:
        for var in costs_vars:
            cost_values.append(-1)
    obj = None
    try:
        obj = m.ObjVal
    except:
        obj = None
   
    
    cost_labels = ["fixed_prod_cost","variable_prod_cost","inv_hold_cost","routing_cost","recourse_cost","unused_routing_cost","expected_cost_diff"]
    if VARIABLE_COST>0:
        results.append([file_path.split("/")[-1].strip(".dat"),m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime+construction_runtime ,sorted(used_vehicles),Q,error,recourse_gamma,variable_crowd_factor,fixed_own,fixed_crowd,r_payment,VARIABLE_COST, BINOM, binom_p]+cost_values)
        column_labels = ["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recourse_gamma","variable_crowd_factor","fixed_own","fixed_crowd","r_payment","VAR_COST_TYPE","binom_dist","binom_p"]+cost_labels
    else:
        results.append([file_path.split("/")[-1].strip(".dat"),m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime+construction_runtime ,sorted(used_vehicles),Q,error,recourse_gamma]+cost_values)
        column_labels = ["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recourse_gamma"]+cost_labels
    for key in m._added_cuts:
        results[-1].append(len(m._added_cuts[key]))
        column_labels.append(key)
    df = pd.DataFrame(results, columns=column_labels)
    df.to_csv("results/model1_homogeneousK_16_threads_integral_Med2_CrodwDrivers_determ_TorresTest_vss_Calc_twoFleets_diffQTest_5Vehicles.csv")
    env.close()

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
mainFolderPath = "./Instances/MVPRP_Dataset_Rev/"

#files = [mainFolderPath+"MVPRP_C15_P6_V3_I3.dat"]



filenames = []
for subdir, dirs, files in os.walk(mainFolderPath):
    for file in files:
        if ("_V3_" in file) and ("_I3" in file or "_I1" in file):
            experiment_folder = subdir.split(os.sep)[-1]
            filepath = os.path.join(subdir, file)
            filenames.append(filepath)

#filenames = [mainFolderPath+"/MVPRP_C15_P6_V3_I3.dat"]
#filenames = ["./Instances/DATA_MVPRP_Rev/MVPRP_C10_P3_V2_I1.dat"]

def compute_vss_fixed_routes(route_info_path):
    #route_info_path ="routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_medPlus2_integral_MIPstart_20251103" #VSS I3
    #route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_integral_MIPstart_vss_routes_rg3_20251103" #VSS I1
    #route_info_path ="routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_medPlus2_integral_MIPstart_vss_routes_rg3_f100_fc50_vc0.5_ri0_20250902"
    #route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_VOD_VSS_Routes_20251113"
    # File pattern to match
    file_pattern = "*_sol.csv"  # Change this to match your files
    
    # Check if folders exist
    print("Folder Check:")
    exists = "✓" if os.path.exists(route_info_path) else "✗"
    print(f"  {exists} : {route_info_path}")
    # Load all folders at once
    
    route_data = load_folder_data_routes(route_info_path, file_pattern)
    return route_data
            
            
#filenames =["./Instances/MVPRP_Dataset_Rev/MVPRP_C15_P3_V3_I3.dat"]
#filenames =["./Instances/Data_Test/MVPRP_C10_P3_V2_I1.dat"]

#filenames =["./Instances/DATA_MVPRP_Rev/MVPRP_C10_P3_V2_I3.dat"]

#filenames =["./Instances/Data_Test/MVPRP_C5_P2_V2_I1.dat"]


#filenames =["./Instances/MVPRP_Dataset_Rev/MVPRP_C10_P3_V3_I1.dat"]

#filenames =["./Instances/Data_Test/MVPRP_C10_P6_V2_I1.dat"]

#filenames =["./Instances/Data_Test/MVPRP_C25_P6_V4_I1.dat"]

#filenames = ["./Instances/Data_Test/MVPRP_C25_P9_V3_I1.dat"]
#filenames =["./Instances/Data_Test/MVPRP_C10_P6_V4_I1.dat"]
#filenames =["./Instances/Data_Test/MVPRP_C15_P9_V3_I1.dat"]



#filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P3_V4_I1.dat"]
#filenames =["./Instances/DATA_MVPRP_Rev/MVPRP_C10_P3_V2_I1.dat"]
#filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C35_P3_V4_I1.dat"]
#filenames = [i for i in filenames if i not in filenames2]

results = []
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


settings = [{"recourse_gamma" : 2.0,
    "fixed_own" : 100,
    "fixed_crowd" : 50,
    "variable_crowd_factor" : 0.5,
    "r_payment" : 0},]
settings = [{"recourse_gamma" : 2.0,
    "fixed_own" : 100,
    "fixed_crowd" : 50,
    "variable_crowd_factor" : 0.5,
    "r_payment" : 0,
    "stoch" : True,
    "modelCut": False,
    "integral": True,
    "MIPstart": False,
    "calc_root" : False,
    "VARIABLE_COST": 2,
    "STOCH_VALUE": True,
    "route_data":None,
    "binom_dist" : True,
    "binom_p": 0.05},]
for setting in settings:
    
    if setting["STOCH_VALUE"]==True:
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_VOD_VSS_Routes_20251113"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_medPlus2_Crowd_Drivers_Bernoulli_20251114"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_medPlus2_integral_MIPstart_CD_MED3_20251117"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_oneCommodityFlow_determ_medPlus2_integral_CD_MED3_VCD_20251118"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_oneCommodityFlow_determ_medPlus2_integral_CD_MED3_VCD_20251118"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_oneCommodityFlow_determ_medPlus2_integral_CD_MED3_VCD_20251121"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_oneCommodityFlow_determ_medPlus2_integral_CD_MED3_VCD_20251122"
        #route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_oneCommodityFlow_determ_medPlus2_integral_CD_MED3_VSSROuteGEn_SameQTorres_20251127"
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_oneCommodityFlow_determ_QHalf_CD_AVG5_VCD_20251201"
        setting["route_data"] = compute_vss_fixed_routes(route_info_path)
    for file_path in filenames:
        run_model(file_path,setting, results= results)
