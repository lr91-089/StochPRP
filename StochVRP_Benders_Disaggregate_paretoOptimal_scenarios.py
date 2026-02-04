#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:27:12 2025

@author: rocha01
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
#import igraph as ig
import glob
from pathlib import Path
import os
import heapq
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import gurobipy as gp
from gurobipy import Model, GRB, quicksum, tuplelist
from cap_sep_mip import separate_fractional_capacity_inequalities
from construction_procedure_benders import generate_initial_solution



stoch = True

modelCut = True

MIPstart = True

calc_root = False#True

pareto = True

SINGLE_CUT = True

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


@dataclass
class PendingSolution:
    """Class to store pending solution data"""
    obj: float
    x_dict: Dict
    z_dict: Dict
    q_dict: Dict
    I_dict: Dict
    y_dict: Dict
    p_dict: Dict
    theta_dict: Dict
    timestamp: float
    cuts_added: bool
    
    def __lt__(self, other):
        """For heap ordering - smaller objective is better"""
        return self.obj < other.obj
    

class SolutionPool:
    """Manages a pool of the best X solutions with duplicate objective prevention"""
    
    def __init__(self, max_size=10, tolerance=1e-6):
        self.max_size = max_size
        self.tolerance = tolerance  # Tolerance for comparing objective values
        self.solutions = []  # Max heap (we'll negate values for min behavior)
        self.solution_count = 0
        self.objectives = set()  # Track existing objective values
    
    def _is_duplicate_objective(self, obj_value: float) -> bool:
        """Check if an objective value already exists in the pool (within tolerance)"""
        for existing_obj in self.objectives:
            if abs(obj_value - existing_obj) <= self.tolerance:
                return True
        return False
    
    def add_solution(self, solution: PendingSolution) -> bool:
        """
        Add a solution to the pool. Returns True if added.
        Prevents adding solutions with duplicate objectives.
        """
        # Check for duplicate objective first
        if self._is_duplicate_objective(solution.obj):
            print(f"Rejected solution - duplicate objective {solution.obj:.6f} already exists in pool")
            return False
        
        if len(self.solutions) < self.max_size:
            # Pool not full, add directly
            heapq.heappush(self.solutions, (-solution.obj, self.solution_count, solution))
            self.objectives.add(solution.obj)
            self.solution_count += 1
            print(f"Added solution to pool (obj: {solution.obj:.2f}), pool size: {len(self.solutions)}")
            return True
        else:
            # Pool is full, check if this solution is better than the worst
            worst_obj = -self.solutions[0][0]  # Negate back to get actual objective
            if solution.obj < worst_obj:
                # Remove worst solution and add new one
                removed = heapq.heappop(self.solutions)
                removed_solution = removed[2]
                self.objectives.remove(removed_solution.obj)  # Remove old objective from set
                
                heapq.heappush(self.solutions, (-solution.obj, self.solution_count, solution))
                self.objectives.add(solution.obj)  # Add new objective to set
                self.solution_count += 1
                print(f"Replaced worst solution (obj: {worst_obj:.2f}) with better solution (obj: {solution.obj:.2f})")
                return True
            else:
                print(f"Solution (obj: {solution.obj:.2f}) not better than worst in pool (obj: {worst_obj:.2f})")
                return False
    
    def get_solutions_sorted(self) -> List[PendingSolution]:
        """Get all solutions sorted by objective (best first)"""
        # Sort by objective value (best first) - x[0] is -obj, so we want reverse=True to get best first
        sorted_solutions = sorted(self.solutions, key=lambda x: x[0], reverse=True)  # x[0] is -obj
        return [sol[2] for sol in sorted_solutions]  # sol[2] is the PendingSolution object
    
    def remove_solutions(self, solutions_to_remove: List[PendingSolution]):
        """Remove specific solutions from the pool"""
        if not solutions_to_remove:
            return
            
        # Create set of solution IDs to remove for efficient lookup
        remove_ids = {id(sol) for sol in solutions_to_remove}
        
        # Keep track of objectives being removed
        removed_objectives = {sol.obj for sol in solutions_to_remove}
        
        # Filter out solutions to remove
        original_size = len(self.solutions)
        self.solutions = [
            (neg_obj, count, sol) for neg_obj, count, sol in self.solutions 
            if id(sol) not in remove_ids
        ]
        
        # Remove the objectives from our tracking set
        for obj in removed_objectives:
            self.objectives.discard(obj)  # Use discard to avoid KeyError if obj not in set
        
        # Re-heapify after removal
        heapq.heapify(self.solutions)
        
        removed_count = original_size - len(self.solutions)
        if removed_count > 0:
            print(f"Removed {removed_count} solutions from pool, remaining: {len(self.solutions)}")
    
    def clear(self):
        """Clear all solutions from the pool"""
        self.solutions.clear()
        self.objectives.clear()
        print("Solution pool cleared")
    
    def size(self) -> int:
        """Get current pool size"""
        return len(self.solutions)
    
    def is_empty(self) -> bool:
        """Check if pool is empty"""
        return len(self.solutions) == 0
    
    def get_objectives(self) -> set:
        """Get set of all objective values in the pool (for debugging)"""
        return self.objectives.copy()
    
    def has_objective(self, obj_value: float) -> bool:
        """Check if a specific objective value exists in the pool"""
        return self._is_duplicate_objective(obj_value)




def solve_recourse_subproblem_for_theta(main_model, x_dict, z_dict, scenario, period):
    """
    More accurate theta calculation by solving the actual recourse subproblem
    (Optional - use this for better theta estimates)
    """
    if scenario not in main_model._submodels:
        print(f"Warning: No submodel found for scenario {scenario}")
        return 0.0
    
    submodel = main_model._submodels[scenario]
    
    try:
        # Solve the dual subproblem for this scenario
        result = solve_dual_subproblem(x_dict, z_dict, submodel)
        obj_val = result[0]
        
        if obj_val is not None:
            return max(0.0, obj_val)
        else:
            print(f"Warning: Subproblem for scenario {scenario}, period {period} did not solve optimally")
            return 0.0
            
    except Exception as e:
        print(f"Error solving subproblem for scenario {scenario}, period {period}: {e}")
        return 0.0

def calculate_theta_values_exact(main_model, construction_model, christofides_routes):
    """
    Calculate exact theta values by solving recourse subproblems
    (More computationally expensive but more accurate)
    """
    print("Calculating exact theta values using subproblem evaluation...")
    
    # Extract solution dictionaries (same as before)
    construction_vars = {var.VarName: var for var in construction_model.getVars()}
    x_dict = {}
    z_dict = {}
    
    # Build solution dictionaries (same extraction logic as above)
    for var_name, var in construction_vars.items():
        if var.x > 1e-6:
            if var_name.startswith('x['):
                indices_str = var_name[2:-1]
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 4:  # x[i,j,t,k]
                    i, j, t, k = indices
                    key = (i, j, t)
                    x_dict[key] = x_dict.get(key, 0) + var.x
                elif len(indices) == 3:  # x[i,j,t]
                    i, j, t = indices
                    x_dict[(i, j, t)] = var.x
                    
            elif var_name.startswith('z['):
                indices_str = var_name[2:-1]
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 3:  # z[i,t,k]
                    i, t, k = indices
                    key = (i, t)
                    z_dict[key] = z_dict.get(key, 0) + var.x
                elif len(indices) == 2:  # z[i,t]
                    i, t = indices
                    z_dict[(i, t)] = var.x

    # Ensure binary values
    z_dict = {k: min(1.0, v) for k, v in z_dict.items()}
    
    # Calculate exact theta values using subproblems
    theta_values = {}
    
    for s in main_model._probabilities:
        if s > 0:  # Skip deterministic scenario
            for t in main_model._periods:
                # Solve the recourse subproblem for exact theta value
                theta_val = solve_recourse_subproblem_for_theta(main_model, x_dict, z_dict, s, t)
                theta_values[(t, s)] = theta_val
                print(f"Exact Theta[{t},{s}] = {theta_val:.4f}")
    
    return theta_values

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
    q_var_mapping = {}
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
            if construction_var.x > 1e-6:
                indices_str = var_name[2:-1]  # Remove 'z[' and ']'
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 3:
                    i, t, k = indices
                    q_var_mapping[(i, t, k)] = round(construction_var.x,0)
                
    # Step 3: Create mappings for routing variables (Model1 has no vehicle index)
    x_var_mapping = {}
    z_var_mapping = {}
    f_var_mapping = {}
    
    
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
            
        elif var_name.startswith('f['):
            try:
                indices_str = var_name[2:-1]  # Remove 'z[' and ']'
                indices = [int(x.strip()) for x in indices_str.split(',')]
                if len(indices) == 3:
                    i, j, t = indices
                    f_var_mapping[(i, j, t)] = var
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
            qf = 0

                        
            for node in route:
                if (node, period) in z_var_mapping:
                    z_var_mapping[(node, period)].start = 1.0
                    used_nodes.add((node, period))
                    route_vars_set += 1
                    if node>0:
                        qf += q_var_mapping[node,period,vehicle]
            
            # Set x variables for edges in the route (without vehicle index for Model1)
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i + 1]
                
                if (node_from, node_to, period) in x_var_mapping:
                    if node_from==0:
                        qi = 0
                    else:
                        qi = q_var_mapping[node_from,period,vehicle]
                    if (node_from, node_to, period) in f_var_mapping:
                        qf = qf-qi
                        f_var_mapping[(node_from, node_to, period)].start = qf
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
            
    # Calculate and set theta values
    theta_values = calculate_theta_values_exact(main_model, construction_model, christofides_routes)
    
    # Set theta variables in the main model
    theta_vars_set = 0
    for (t, s), value in theta_values.items():
        if (t, s) in main_model._theta:
            main_model._theta[t, s].start = value
            theta_vars_set += 1
    if 0 in main_model._probabilities:
        main_model._theta_zero.start = main_model._probabilities[0] * (
            sum(main_model._r[i] * z_var_mapping[(i, t)].start for (i,t) in z_var_mapping if i>0) - 
            sum(main_model._c[i, j] * x_var_mapping[(i, j, t)].start for (i,j,t) in x_var_mapping))
    
    print(f"Set {theta_vars_set} theta variables for MIP start")
    
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
        stoch=stoch,
        #het_model=True
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

def input_ordering(df,depot, nodes):
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
                if len(cut_set)>5:
                    break
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
        model._added_cuts["lsec"].add(S)
        n+=1
        if n>1:
            return found_st
    return found_st

def eliminate_subtours_components(model, edges,q_vars,z_vars, period, mipsol=True):
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
                    z_comp = {key:z_vars.get(key,0) for key in comp}
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
                    return True

    return False


def get_tours(x_arcs,probability):
    tours = []
    for arcs in x_arcs:
        if arcs[0]==0:   
            i = 0
            j = arcs[1]
            sorted_tour = [(i,j)]
            customer_set = set((j,))
            while j!=0:
                for e in x_arcs:
                    if e[0]==j:
                        i = e[0]
                        if j!=0:
                            customer_set.add(j)



def calculate_scaling_factor(lambda1, lambda2, lambda3,lambda6,lambda7):
    """Calculate L2-norm across all dual multipliers for cut scaling"""
    # Collect all coefficient values from all dual variables
    all_coefficients = []
    
    # Add values from each set of dual multipliers
    all_coefficients.extend([abs(val) for key, val in lambda1.items()])
    all_coefficients.extend([abs(val) for key, val in lambda2.items()])
    all_coefficients.extend([abs(val) for key, val in lambda3.items()])
    all_coefficients.extend([abs(val) for key, val in lambda6.items()])
    all_coefficients.extend([abs(val) for key, val in lambda7.items()])
    
    # Calculate the L2-norm (Euclidean norm)
    l2_norm = math.sqrt(sum(x*x for x in all_coefficients))
    
    # Avoid division by zero or very small numbers
    return max(l2_norm, 1e-9)

def get_tours(model,x_arcs,probability):
    tours = []
    for arcs in x_arcs:
        if arcs[0]==0:   
            i = 0
            j = arcs[1]
            sorted_tour = [(i,j)]
            customer_set = set((j,))
            while j!=0:
                for e in x_arcs:
                    if e[0]==j:
                        i = e[0]
                        if j!=0:
                            customer_set.add(j)
                        j = e[1]
                        sorted_tour.append(e)
                        break
            tours.append(((sorted_tour),model._customer_set,sum(model._c[e] for e in sorted_tour)*probability+(1-probability)*sum(model._r[i] for i in model._customer_set)))
    return tours

def check_tour_capacity(x_arcs,q_arcs, period,Q, error):
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


def initialize_core_point(model,x_dict,z_dict):
    """Initialize core point according to Proposition 1 in the paper"""
    core_point = {}
    core_point['x'] = {k: 0.5 * v for k, v in x_dict.items()}
    core_point['z'] = {k: 0.5 * v for k, v in z_dict.items()}
    return core_point
    
def update_core_point(current_core_point, x_values, z_values, alpha=0.5):
    # Defensive programming: default to 0 if keys not found
    def blend(a, b): return alpha * a + (1 - alpha) * b
    
    updated_core_point = {
        'x': {k: blend(current_core_point['x'].get(k, 0.0), x_values.get(k, 0.0)) 
              for k in set(current_core_point['x']) | set(x_values)},
        'z': {k: blend(current_core_point['z'].get(k, 0.0), z_values.get(k, 0.0)) 
              for k in set(current_core_point['z']) | set(z_values)}
    }
    return updated_core_point

def compute_theta_values_for_solution_updated(model, x_dict, z_dict, theta_dict,sol_obj,cur_ub):
    """
    Compute the correct theta values for a given x,z solution
    Updated for period-scenario theta structure: theta[t,s]
    """
    
    probabilities = []
    
    for s in model._probabilities:
        for t in model._periods:
            if (t,s) not in theta_dict:
                probabilities.append(s)
                
    # Compute theta for each period and scenario combination
    for s in probabilities:
        if s > 0:  # Skip deterministic scenario (s=0) if it exists
            # Solve the recourse subproblem for this scenario
            submodel = model._submodels[s]
            dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7  = solve_dual_subproblem(x_dict, z_dict, submodel)
            sol_obj += dObj
            if sol_obj>cur_ub:
                return [], False
            if dObj is not None:
                for t in model._periods:
                    try:
                        # The theta value should be at least the recourse cost
                        # Note: this might need to be adjusted based on your specific model structure
                        #theta_val = dObj   # Divide by periods if aggregated
                        check = 0.0
                        check += model._probabilities[s] * (
                            sum(model._r[i] * z_dict[i, t] for i in model._customers) - 
                            sum(model._c[i, j] * x_dict[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                        )
                        check += -lambda1.get((t, s), 0) * s
                        check += -sum(lambda2.get((i, j, t, s), 0) * x_dict[i, j, t] 
                                        for i in model._nodes for j in model._nodes if i != j)
                        check += -sum(lambda3.get((i, t, s), 0) * z_dict[i, t] for i in model._customers)
                        check += -sum(lambda6[i,j,t,s] for (i,j) in model._c)
                        check += -sum(lambda7[i,t,s] for i in model._customers)
                        theta_dict[(t, s)] = abs(round(check,4))
                        
                    except Exception as e:
                        print(f"Error computing theta for scenario {s}, period {t}: {e}")
                        theta_dict[(t, s)] = 1e6
    
    return theta_dict, True

def compute_theta_zero_value_updated(model, x_dict, z_dict):
    """
    Compute theta_0 value for deterministic scenario if it exists
    Updated for your model structure
    """
    if 0 in model._probabilities:
        theta_zero = model._probabilities[0] * (
            sum(model._r[i] * z_dict.get((i, t), 0) for i in model._customers for t in model._periods) - 
            sum(model._c[i, j] * x_dict.get((i, j, t), 0) for i in model._nodes for j in model._nodes if i != j for t in model._periods)
        )
        return max(0.0, theta_zero)
    return 0.0



def try_inject_solution(model, solution: PendingSolution, cur_ub: float) -> tuple[bool, Optional[float]]:
    """
    Try to inject a single solution. Returns (success, new_obj)
    """
    try:
        print(f"Attempting injection of solution with obj {solution.obj:.2f}")
        
        # Compute correct theta values for this solution
        theta_values, sol_better = compute_theta_values_for_solution_updated(
            model, solution.x_dict, solution.z_dict,solution.theta_dict,solution.obj,cur_ub)
        theta_zero_value = compute_theta_zero_value_updated(
            model, solution.x_dict, solution.z_dict)
        
        if sol_better==False:
            return False, None
        # Check with computed theta values
        total_obj = solution.obj + sum(theta_values.values())
        if hasattr(model, '_theta_zero'):
            total_obj += theta_zero_value
            
        if total_obj + 0.001 >= cur_ub:
            print(f"  Rejected - total obj {total_obj:.2f} >= UB {cur_ub:.2f}")
            return False, None
        
        # Prepare variable lists for cbSetSolution
        solution_vars = []
        solution_vals = []
        
        # Add all original variables
        for key, var in model._x.items():
            solution_vars.append(var)
            solution_vals.append(solution.x_dict.get(key, 0.0))
            
        for key, var in model._z.items():
            solution_vars.append(var)
            solution_vals.append(solution.z_dict.get(key, 0.0))
            
        for key, var in model._q.items():
            solution_vars.append(var)
            solution_vals.append(solution.q_dict.get(key, 0.0))
            
        for key, var in model._I.items():
            solution_vars.append(var)
            solution_vals.append(solution.I_dict.get(key, 0.0))
            
        for key, var in model._y.items():
            solution_vars.append(var)
            solution_vals.append(solution.y_dict.get(key, 0.0))
            
        for key, var in model._p.items():
            solution_vars.append(var)
            solution_vals.append(solution.p_dict.get(key, 0.0))
        
        # Add theta variables with computed values
        theta_vars_set = 0
        for (t, s), theta_val in theta_values.items():
            if (t, s) in model._theta:
                solution_vars.append(model._theta[t, s])
                solution_vals.append(theta_val)
                theta_vars_set += 1
        
        # Add theta_zero if it exists
        if hasattr(model, '_theta_zero'):
            solution_vars.append(model._theta_zero)
            solution_vals.append(theta_zero_value)
        
        # Inject the solution
        model.cbSetSolution(solution_vars, solution_vals)
        new_obj = model.cbUseSolution()
        
            
        if new_obj < model._best_feasible_obj:
            model._best_feasible_obj = new_obj
        
        print(f"  Successfully injected solution with obj {new_obj:.2f} (theta vars: {theta_vars_set})")
        if theta_zero_value > 0:
            print(f"  Theta_zero value: {theta_zero_value:.2f}")
        
        return True, new_obj
        
    except Exception as e:
        print(f"  Error injecting solution: {e}")
        return False, None
    
    
def do_multi_cut(model, x_dict, z_dict, all_q_vars, curr_obj, solcnt, UB):
    total_violation = 0.0
    theta_vals = {}
    add_benders_cuts = True
    for s in model._probabilities:
        if s > 0:
            alllambda1 = {}
            alllambda2 = {}
            alllambda3 = {}
            alllambda6 = {}
            alllambda7 = {}
            
            # Sum theta over all periods for this scenario
            theta = 0
            for t in model._periods:
                theta += model.cbGetSolution(model._theta[t,s])
                
            if pareto==True:
                dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = generate_pareto_optimal_cut(
                    model, s, t, x_dict, z_dict, theta, model._core_point)
            else:
                dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = solve_dual_subproblem(x_dict, z_dict, model._submodels[s])
            
            alllambda1 = alllambda1 | lambda1
            alllambda2 = alllambda2 | lambda2
            alllambda3 = alllambda3 | lambda3
            alllambda6 = alllambda6 | lambda6
            alllambda7 = alllambda7 | lambda7
            
            if dObj is not None and dObj > pow(10, -6):
                # Add cuts for all periods with this scenario
                for t in model._periods:
                    expr = 0.0
                    check = 0.0
                    expr += model._probabilities[s] * (
                        quicksum(model._r[i] * model._z[i, t] for i in model._customers) - 
                        quicksum(model._c[i, j] * model._x[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                    )
                    expr += -lambda1[t, s] * s
                    expr += -quicksum(lambda2[i, j, t, s] * model._x[i, j, t] 
                                    for i in model._nodes for j in model._nodes if i != j)
                    expr += -quicksum(lambda3[i, t, s] * model._z[i, t] for i in model._customers)
                    expr += -quicksum(lambda6[i,j,t,s] for (i,j) in model._c)
                    expr += -quicksum(lambda7[i,t,s] for i in model._customers)
                    
                    check += model._probabilities[s] * (
                        sum(model._r[i] * z_dict[i, t] for i in model._customers) - 
                        sum(model._c[i, j] * x_dict[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                    )
                    check += -lambda1.get((t, s), 0) * s
                    check += -sum(lambda2.get((i, j, t, s), 0) * x_dict[i, j, t] 
                                    for i in model._nodes for j in model._nodes if i != j)
                    check += -sum(lambda3.get((i, t, s), 0) * z_dict[i, t] for i in model._customers)
                    check += -sum(lambda6[i,j,t,s] for (i,j) in model._c)
                    check += -sum(lambda7[i,t,s] for i in model._customers)
                    
                    theta_val = model.cbGetSolution(model._theta[t,s])
                    violation = check - theta_val
                    theta_vals[t,s] = abs(round(check,4))
                    if violation > 1e-4:
                        if add_benders_cuts == True:
                            total_violation += violation
                            cuts_added_this_callback = True
                            cuts_added = True
                            # CRITICAL: Just store the solution data, don't call cbUseSolution here
                            # We'll inject the corrected solution in MIPNODE callback
                            # Check if this is a new best solution
                            
                            # Store cut data
                            if cuts_added_this_callback:    
                                q_dict = {e: all_q_vars[e] for e in all_q_vars}
                                all_I_vars = model.cbGetSolution(model._I)
                                all_y_vars = model.cbGetSolution(model._y)
                                all_p_vars = model.cbGetSolution(model._p)
                                I_dict = {e: all_I_vars[e] for e in all_I_vars}
                                y_dict = {e: all_y_vars[e] for e in all_y_vars}
                                p_dict = {e: all_p_vars[e] for e in all_p_vars}
                                model._benders_cuts.append({
                                    'dopt': dObj,
                                    's': s,
                                    'x': x_dict,
                                    'z': z_dict,
                                    'q': q_dict,
                                    'I': I_dict,
                                    'y': y_dict,
                                    'p': p_dict
                                })
                            
                            print(f"Adding optimality cut theta{(t,s)} with violation {violation}, incumbent obj {curr_obj}, check sum of cut {check}")
                            
                            # Add the Benders cut - UPDATED for (t,s) indexing
                            model._theta_scale[t,s] = calculate_scaling_factor(alllambda1, alllambda2, alllambda3, alllambda6, alllambda7)
                            expr = (1/model._theta_scale[t,s]) * expr
                            model.cbLazy(expr <= (1/model._theta_scale[t,s]) * model._theta[t,s])
                        if solcnt > 0 and curr_obj+total_violation>UB:
                            add_benders_cuts = False
        return add_benders_cuts

def cb(model, where):
    
    if where == GRB.Callback.MIPSOL:
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        UB = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        if UB<model._best_feasible_obj:
            model._best_feasible_obj = UB
        LB = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        curr_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        gap = (UB-LB)/UB if UB > 0 else float('inf')
        

        
        # Get the current solution values
        all_x_vars = model.cbGetSolution(model._x)
        all_z_vars = model.cbGetSolution(model._z)
        all_q_vars = model.cbGetSolution(model._q)
        epsilon = pow(10, -4)
        
        # Filter significant values and create dictionaries
        x_arcs = {a: all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
        x_dict = {}
        x_sparse = {}
        for e in all_x_vars:
            if all_x_vars[e] > 0.5:
                x_dict[e] = 1.0
                x_sparse[e] = 1.0
            else:
                x_dict[e] = 0.0
        z_dict = {}
        z_sparse = {}
        for e in all_z_vars:
            if all_z_vars[e] > 0.5:
                z_dict[e] = 1.0
                z_sparse[e] = 1.0
            else:
                z_dict[e] = 0.0
        
        # First check for subtours/capacity cuts
        cuts_added = False
        if modelCut==True:
            for t in random.sample(model._periods,len(model._periods)):
                x_arcs_dict = {(a[0], a[1]): {"capacity": x_arcs[a]} for a in x_arcs if a[2] == t}
                q_vars = {a[0]: all_q_vars[a] for a in all_q_vars if (all_q_vars[a] > epsilon and a[1] == t)}
                z_vars = {a[0]: all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1] == t)}
                
                # Check for subtour elimination constraints
                start = time.time()
                found_sec = eliminate_subtours_components(model, x_arcs_dict, q_vars, z_vars, t)
                model._cb_times["l_st_time"].append(time.time()-start)
                if found_sec:
                    cuts_added = True
                    # Store the solution with computed theta values before returning
                    #if is_new_best:
                        #store_complete_solution_updated(model)
                    return
                
                # Check for capacity cuts
                start = time.time()
                
                if separate_fractional_capacity_inequalities(model, x_arcs_dict, z_vars, q_vars, model._Q, model._num_nodes, t, mipsol=True):
                    cuts_added = True
                    # Store the solution with computed theta values before returning
                    #if is_new_best:
                        #store_complete_solution_updated(model)
                    return
                model._cb_times["l_fcc_time"].append(time.time()-start)
        
        is_new_best = False

        # Now check for Benders cuts
        start = time.time()
        cuts_added_this_callback = False
        
        # Initialize core point if needed
        if model._core_point == None:
            model._core_point = initialize_core_point(model, x_dict, z_dict)
        model._core_point = update_core_point(model._core_point, x_dict, z_dict)
        total_violation = 0.0
        theta_vals = {}
        add_benders_cuts = True
        if SINGLE_CUT==True:
            alllambda1 = {}
            alllambda2 = {}
            alllambda3 = {}
            alllambda6 = {}
            alllambda7 = {}
            
            # Sum theta over all periods for this scenario
            theta = 0
            #for s in model._probabilities:
             #   for t in model._periods:
              #      theta += model.cbGetSolution(model._theta[t,s])
                
            if pareto==True:
                dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = generate_pareto_optimal_cut(
                    model, -1, x_dict, z_dict, theta, model._core_point, single_cut=True)
            else:
                dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = solve_dual_subproblem_single_cut(x_dict, z_dict, model._submodel)
            
            alllambda1 = alllambda1 | lambda1
            alllambda2 = alllambda2 | lambda2
            alllambda3 = alllambda3 | lambda3
            alllambda6 = alllambda6 | lambda6
            alllambda7 = alllambda7 | lambda7
            
            if dObj is not None:
                for s in model._probabilities:
                    if s > 0:
                      theta = 0
                      #for s in model._probabilities:
                      for t in model._periods:
                            theta += model.cbGetSolution(model._theta[t,s])
                      # Add cuts for all periods with this scenario
                      for t in model._periods:
                          expr = 0.0
                          check = 0.0
                          expr += model._probabilities[s] * (
                              quicksum(model._r[i] * model._z[i, t] for i in model._customers) - 
                              quicksum(model._c[i, j] * model._x[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                          )
                          expr += -lambda1[t, s] * s
                          expr += -quicksum(lambda2[i, j, t, s] * model._x[i, j, t] 
                                          for i in model._nodes for j in model._nodes if i != j)
                          expr += -quicksum(lambda3[i, t, s] * model._z[i, t] for i in model._customers)
                          expr += -quicksum(lambda6[i,j,t,s] for (i,j) in model._c)
                          expr += -quicksum(lambda7[i,t,s] for i in model._customers)
                          
                          check += model._probabilities[s] * (
                              sum(model._r[i] * z_dict[i, t] for i in model._customers) - 
                              sum(model._c[i, j] * x_dict[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                          )
                          check += -lambda1.get((t, s), 0) * s
                          check += -sum(lambda2.get((i, j, t, s), 0) * x_dict[i, j, t] 
                                          for i in model._nodes for j in model._nodes if i != j)
                          check += -sum(lambda3.get((i, t, s), 0) * z_dict[i, t] for i in model._customers)
                          check += -sum(lambda6[i,j,t,s] for (i,j) in model._c)
                          check += -sum(lambda7[i,t,s] for i in model._customers)
                          
                          theta_val = model.cbGetSolution(model._theta[t,s])
                          violation = abs(check - theta_val)
                          theta_vals[t,s] = abs(round(check,4))
                          if violation > 1e-4:
                              if add_benders_cuts == True:
                                  total_violation += violation
                                  cuts_added_this_callback = True
                                  cuts_added = True
                                  # CRITICAL: Just store the solution data, don't call cbUseSolution here
                                  # We'll inject the corrected solution in MIPNODE callback
                                  # Check if this is a new best solution
                                  
                                  # Store cut data
                                  if cuts_added_this_callback:    
                                      q_dict = {e: all_q_vars[e] for e in all_q_vars}
                                      all_I_vars = model.cbGetSolution(model._I)
                                      all_y_vars = model.cbGetSolution(model._y)
                                      all_p_vars = model.cbGetSolution(model._p)
                                      I_dict = {e: all_I_vars[e] for e in all_I_vars}
                                      y_dict = {e: all_y_vars[e] for e in all_y_vars}
                                      p_dict = {e: all_p_vars[e] for e in all_p_vars}
                                      model._benders_cuts.append({
                                          'dopt': dObj,
                                          's': s,
                                          'x': x_dict,
                                          'z': z_dict,
                                          'q': q_dict,
                                          'I': I_dict,
                                          'y': y_dict,
                                          'p': p_dict
                                      })
                                  
                                  print(f"Adding optimality cut theta{(t,s)} with violation {violation}, incumbent obj {curr_obj}, check sum of cut {check}")
                                  
                                  # Add the Benders cut - UPDATED for (t,s) indexing
                                  model._theta_scale[t,s] = calculate_scaling_factor(alllambda1, alllambda2, alllambda3, alllambda6, alllambda7)
                                  expr = (1/model._theta_scale[t,s]) * expr
                                  model.cbLazy(expr <= (1/model._theta_scale[t,s]) * model._theta[t,s])
                              if solcnt > 0 and curr_obj+total_violation>UB:
                                  add_benders_cuts = False
                             
                          if add_benders_cuts ==False:
                                if curr_obj+total_violation < UB:
                                    is_new_best = True
                                model._cb_times["l_benders"].append(time.time()-start)
                                if is_new_best:
                                     # Store the solution data for later injection in MIPNODE
                                     # Create solution object
                                     solution = PendingSolution(
                                         obj=curr_obj,
                                         x_dict=x_dict.copy(),
                                         z_dict=z_dict.copy(),
                                         q_dict={e: round(all_q_vars[e],0) for e in all_q_vars},
                                         I_dict={e: round(model.cbGetSolution(model._I)[e],0) for e in model._I},
                                         y_dict={e: round(model.cbGetSolution(model._y)[e],0) for e in model._y},
                                         p_dict={e: round(model.cbGetSolution(model._p)[e],0) for e in model._p},
                                         theta_dict = theta_vals,
                                         timestamp=time.time(),
                                         cuts_added=cuts_added
                                     )
                                     
                                     # Add to solution pool
                                     added = model._solution_pool.add_solution(solution)
                  
                                return
        else:
            for s in model._probabilities:
                if s > 0:
                    alllambda1 = {}
                    alllambda2 = {}
                    alllambda3 = {}
                    alllambda6 = {}
                    alllambda7 = {}
                    
                    # Sum theta over all periods for this scenario
                    theta = 0
                    for t in model._periods:
                        theta += model.cbGetSolution(model._theta[t,s])
                        
                    if pareto==True:
                        dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = generate_pareto_optimal_cut(
                            model, s, x_dict, z_dict, theta, model._core_point)
                    else:
                        dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = solve_dual_subproblem(x_dict, z_dict, model._submodels[s])
                    
                    alllambda1 = alllambda1 | lambda1
                    alllambda2 = alllambda2 | lambda2
                    alllambda3 = alllambda3 | lambda3
                    alllambda6 = alllambda6 | lambda6
                    alllambda7 = alllambda7 | lambda7
                    
                    if dObj is not None:# and dObj > pow(10, -6):
                        # Add cuts for all periods with this scenario
                        for t in model._periods:
                            expr = 0.0
                            check = 0.0
                            expr += model._probabilities[s] * (
                                quicksum(model._r[i] * model._z[i, t] for i in model._customers) - 
                                quicksum(model._c[i, j] * model._x[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                            )
                            expr += -lambda1[t, s] * s
                            expr += -quicksum(lambda2[i, j, t, s] * model._x[i, j, t] 
                                            for i in model._nodes for j in model._nodes if i != j)
                            expr += -quicksum(lambda3[i, t, s] * model._z[i, t] for i in model._customers)
                            expr += -quicksum(lambda6[i,j,t,s] for (i,j) in model._c)
                            expr += -quicksum(lambda7[i,t,s] for i in model._customers)
                            
                            check += model._probabilities[s] * (
                                sum(model._r[i] * z_dict[i, t] for i in model._customers) - 
                                sum(model._c[i, j] * x_dict[i, j, t] for i in model._nodes for j in model._nodes if i != j)
                            )
                            check += -lambda1.get((t, s), 0) * s
                            check += -sum(lambda2.get((i, j, t, s), 0) * x_dict[i, j, t] 
                                            for i in model._nodes for j in model._nodes if i != j)
                            check += -sum(lambda3.get((i, t, s), 0) * z_dict[i, t] for i in model._customers)
                            check += -sum(lambda6[i,j,t,s] for (i,j) in model._c)
                            check += -sum(lambda7[i,t,s] for i in model._customers)
                            
                            theta_val = model.cbGetSolution(model._theta[t,s])
                            violation = abs(check - theta_val)
                            theta_vals[t,s] = abs(round(check,4))
                            if violation > 1e-4:
                                if add_benders_cuts == True:
                                    total_violation += violation
                                    cuts_added_this_callback = True
                                    cuts_added = True
                                    # CRITICAL: Just store the solution data, don't call cbUseSolution here
                                    # We'll inject the corrected solution in MIPNODE callback
                                    # Check if this is a new best solution
                                    
                                    # Store cut data
                                    if cuts_added_this_callback:    
                                        q_dict = {e: all_q_vars[e] for e in all_q_vars}
                                        all_I_vars = model.cbGetSolution(model._I)
                                        all_y_vars = model.cbGetSolution(model._y)
                                        all_p_vars = model.cbGetSolution(model._p)
                                        I_dict = {e: all_I_vars[e] for e in all_I_vars}
                                        y_dict = {e: all_y_vars[e] for e in all_y_vars}
                                        p_dict = {e: all_p_vars[e] for e in all_p_vars}
                                        model._benders_cuts.append({
                                            'dopt': dObj,
                                            's': s,
                                            'x': x_dict,
                                            'z': z_dict,
                                            'q': q_dict,
                                            'I': I_dict,
                                            'y': y_dict,
                                            'p': p_dict
                                        })
                                    
                                    print(f"Adding optimality cut theta{(t,s)} with violation {violation}, incumbent obj {curr_obj}, check sum of cut {check}")
                                    
                                    # Add the Benders cut - UPDATED for (t,s) indexing
                                    model._theta_scale[t,s] = calculate_scaling_factor(alllambda1, alllambda2, alllambda3, alllambda6, alllambda7)
                                    expr = (1/model._theta_scale[t,s]) * expr
                                    model.cbLazy(expr <= (1/model._theta_scale[t,s]) * model._theta[t,s])
                                if solcnt > 0 and curr_obj+total_violation>UB:
                                    add_benders_cuts = False
                               
                    if add_benders_cuts ==False:
                        if curr_obj+total_violation < UB:
                            is_new_best = True
                        model._cb_times["l_benders"].append(time.time()-start)
                        if is_new_best:
                             # Store the solution data for later injection in MIPNODE
                             # Create solution object
                             solution = PendingSolution(
                                 obj=curr_obj,
                                 x_dict=x_dict.copy(),
                                 z_dict=z_dict.copy(),
                                 q_dict={e: round(all_q_vars[e],0) for e in all_q_vars},
                                 I_dict={e: round(model.cbGetSolution(model._I)[e],0) for e in model._I},
                                 y_dict={e: round(model.cbGetSolution(model._y)[e],0) for e in model._y},
                                 p_dict={e: round(model.cbGetSolution(model._p)[e],0) for e in model._p},
                                 theta_dict = theta_vals,
                                 timestamp=time.time(),
                                 cuts_added=cuts_added
                             )
                             
                             # Add to solution pool
                             added = model._solution_pool.add_solution(solution)
    
                        return
        if curr_obj+total_violation < UB:
            is_new_best = True

        if is_new_best:
             # Store the solution data for later injection in MIPNODE
             # Create solution object
             solution = PendingSolution(
                 obj=curr_obj,
                 x_dict=x_dict.copy(),
                 z_dict=z_dict.copy(),
                 q_dict={e: round(all_q_vars[e],0) for e in all_q_vars},
                 I_dict={e: round(model.cbGetSolution(model._I)[e],0) for e in model._I},
                 y_dict={e: round(model.cbGetSolution(model._y)[e],0) for e in model._y},
                 p_dict={e: round(model.cbGetSolution(model._p)[e],0) for e in model._p},
                 theta_dict = theta_vals,
                 timestamp=time.time(),
                 cuts_added=cuts_added
             )
             # Add to solution pool
             added = model._solution_pool.add_solution(solution)
             
        #if curr_obj<52070.0:
            #breakpoint()
            #print("error sol")
        model._cb_times["l_benders"].append(time.time()-start)
   
    # MIPNODE callback - this is where we try to inject solutions from the pool
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        cur_ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
        nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        
        # Check if we have solutions in the pool and enough time has passed

        if not model._solution_pool.is_empty():
            current_time = time.time()
            
            # Get the best solution from pool to check age
            solutions = model._solution_pool.get_solutions_sorted()
            oldest_solution = min(solutions, key=lambda s: s.timestamp)
            solution_age = current_time - oldest_solution.timestamp
            
            # Only try injection if oldest solution has been pending for at least 10 seconds
            if solution_age >= 10.0:
                print(f"Attempting to inject solutions from pool at node {nodecnt} (pool size: {model._solution_pool.size()})")
                
                success = False
                solutions_to_try = model._solution_pool.get_solutions_sorted()
                solutions_checked = []
                
                # Try each solution in order of quality until one succeeds
                for i, solution in enumerate(solutions_to_try):
                    # Quick check: skip if base objective is already worse than current UB
                    if solution.obj >= cur_ub - 0.001:
                       print(f"    Skipped - base obj {solution.obj:.2f} >= UB {cur_ub:.2f}")
                       solutions_checked.append(solution)  # Mark as checked
                       continue
                    print(f"  Trying solution {i+1}/{len(solutions_to_try)} (obj: {solution.obj:.2f})")
                    
                    injection_success, new_obj = try_inject_solution(model, solution, cur_ub)
                    solutions_checked.append(solution)  # Mark as checked regardless of success
                    
                    if injection_success:
                        print(f"  Successfully injected solution {i+1} with final obj {new_obj:.2f}")
                        success = True
                        break
                    else:
                        print(f"  Solution {i+1} injection failed")
                
                # Remove only the solutions that were checked
                model._solution_pool.remove_solutions(solutions_checked)
                
                if success:
                    print(f"Pool injection successful, removed {len(solutions_checked)} checked solutions")
                else:
                    print(f"No solutions could be injected, removed {len(solutions_checked)} checked solutions")
            #else:
                # Print status occasionally
             #   if nodecnt % 100 == 0:
              #      remaining_wait = 10.0 - solution_age
               #     best_obj = solutions[0].obj if solutions else "N/A"
                #    print(f"Pool injection pending in {remaining_wait:.1f}s (best obj: {best_obj}, pool size: {model._solution_pool.size()})")

        if status == GRB.OPTIMAL:
            cur_lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            
            # Continue with your existing separation logic
            all_x_vars = model.cbGetNodeRel(model._x)
            all_z_vars = model.cbGetNodeRel(model._z)
            all_q_vars = model.cbGetNodeRel(model._q)
            epsilon = pow(10, -4)
            x_arcs = {a:all_x_vars[a] for a in all_x_vars if all_x_vars[a] > epsilon}
            
            if ((nodecnt<1 and len(model._added_cuts["ufcc"])+len(model._added_cuts["usec"])<250) or 
                nodecnt<200 or (nodecnt>200 and nodecnt - model._cb_lastnode >= 400)):
                sep = True
                if (nodecnt>200):
                    if model._cb_last_lower_bound<cur_lb-epsilon or model._cb_last_ub-epsilon>cur_ub:
                        sep = False
                model._cb_last_lower_bound = cur_lb
                model._cb_last_ub = cur_ub
                model._cb_lastnode= nodecnt
                if sep==True:
                    ncut = 0
                    for t in random.sample(model._periods,len(model._periods)):
                        x_arcs_dict = {(a[0],a[1]):{"capacity":x_arcs[a]} for a in x_arcs if a[2]==t}
                        q_vars = {a[0]:all_q_vars[a] for a in all_q_vars if (all_z_vars[a] > epsilon and a[1]==t)}
                        z_vars = {a[0]:all_z_vars[a] for a in all_z_vars if (all_z_vars[a] > epsilon and a[1]==t)}
                        
                        start = time.time()
                        #found_sec = exact_subtour_elemination(model,x_arcs_dict,z_vars,model._n, t, mipsol=False)
                        found_sec = eliminate_subtours_components(model, x_arcs_dict,q_vars, z_vars,t, mipsol=False)
                        model._cb_times["u_st_time"].append(time.time()-start)
                        if modelCut==True:
                            if found_sec==False:
                                start = time.time()
                                found_fcc = separate_fractional_capacity_inequalities(model,x_arcs_dict,z_vars,q_vars,model._Q,model._num_nodes,t)
                                model._cb_times["u_fcc_time"].append(time.time()-start)
                                if found_fcc == True:
                                    ncut+=1
                            else:
                                ncut+=1
                        if ncut > 9:
                            return
    

def build_master_problem_for_benders(dataFile,recourse_gamma, stoch=True):
    """
    Extract your model building code into this function
    This should build the master problem without the callback
    """
    
    num_nodes,num_vehicles, num_days, Q = dataFile["num_nodes"],dataFile["num_vehicles"],dataFile["periods"],dataFile["Q"]
    
    depot = dataFile["supplier"]
    
    # Read remaining rows as customers
    customers = dataFile["retailers"]
        
    nodes = [*range(num_nodes)]

    # Convert to DataFrame
    customers_df = pd.DataFrame(customers)
    
    #customers_df = input_ordering(customers_df,depot, nodes)
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
        num_vehicles = num_vehicles+2
    else:
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
        return float(np.round(math.sqrt(diff[0]*diff[0]+diff[1]*diff[1]),0))

    
    dist = {(c1, c2): distance(c1, c2) for c1 in nodes for c2 in nodes if c1!=c2}
    
    #recourse costs
    r = {}
    for i in customers:
        r[i] = dist[0,i]*recourse_gamma
    
    c = dist
    
    # Create the model object m
    env = gp.Env()
    if pareto==True:
        par_str="_paretoOpt_"
    else:
        par_str="_"
    model_name = f'benders_bc_disaggregate_MW{par_str}delayedPool_rg{recourse_gamma}'
    if MIPstart==True:
        model_name += "_MIPstart"
    m = gp.Model(model_name, env=env)
    
    # Edge variables = 1, if customer i is visited after h in period t by vehicke k
    var_x = m.addVars(c, periods, vtype=GRB.BINARY, name='x')
    #binary variable = 1 if setup occurs in period t
    var_y = m.addVars(periods, vtype=GRB.BINARY, name='y')
    #quantity produced in period t
    var_p = m.addVars(periods,lb=0.0, vtype=GRB.CONTINUOUS, name='p')
    #quantity of inventory at node i in period t
    var_I = m.addVars(nodes,periodsI,lb=0.0, vtype=GRB.CONTINUOUS, name='I')
    #quantity delivered to cust i in period t
    var_q = m.addVars(nodes,periods,lb=0.0, vtype=GRB.CONTINUOUS, name='q')
    #binary variable equal to 1 if node i in N is visited in period t
    var_z = m.addVars(nodes,periods, vtype=GRB.BINARY, name='z')
    if modelCut==False:
        var_f = m.addVars(c,periods,lb=0.0, vtype=GRB.CONTINUOUS, name='f')
    #number of vehicles that leave the production plant in period t
    #var_k = m.addVars(periods,lb=0.0,ub=len(vehicles), vtype=GRB.CONTINUOUS, name='k')
    #continous variable alpha
    prob_pos = {s:probabilities[s] for s in probabilities if s>0}
    if SINGLE_CUT==False:
        var_theta = m.addVars(periods,prob_pos,lb=-100000, vtype=GRB.CONTINUOUS, name="theta")
    else:
        var_theta = m.addVars(periods,prob_pos,lb=-100000, vtype=GRB.CONTINUOUS, name="theta")
        #var_theta = m.addVars(lb=-100000, vtype=GRB.CONTINUOUS, name="theta")
    
    
    
    
    
    m.modelSense = GRB.MINIMIZE
    fixed_production_costs = quicksum(setup_prod_cost*var_y[t] for t in periods)
    variable_production_costs = quicksum(var_prod_cost*var_p[t] for t in periods)
    inventory_holding_costs = quicksum(holding_costs[i]*var_I[i,t] for i in nodes for t in periods)
    routing_costs = quicksum(c[i,j] * var_x[i, j,t] for i in nodes for j in nodes if i!=j for t in periods)
    benders_cost = quicksum(var_theta[t,s] for s in prob_pos for t in periods)
    #if SINGLE_CUT==False:
    if 0 in probabilities:
        var_theta_zero = m.addVar(lb=-100000,vtype=GRB.CONTINUOUS, name="theta_0")
        benders_cost += var_theta_zero 
    m.setObjective(fixed_production_costs +variable_production_costs+inventory_holding_costs+routing_costs+benders_cost)
    
    m.addConstrs((var_I[0,t-1]+var_p[t] == var_I[0,t]+quicksum(var_q[i,t] for i in nodes) for t in periods), name="Inventory balance at plant")
    m.addConstrs((var_I[i,t-1]+var_q[i,t] == var_I[i,t]+daily_demand[i,t] for i in customers for t in periods), name="Inventory balance at customers")
    m.addConstrs((var_I[0,t] <= inv_cap[0] for t in periods), name="Inventory balance at plant")
    m.addConstrs((var_I[i,t-1]+var_q[i,t]  <= inv_cap[i] for i in customers for t in periods), name="Inventory capacity at client after delivery")
    m.addConstrs((var_p[t] <= min(production_capacity[t],sum(daily_demand[i,l] for i in customers for l in periods[t-1:]))*var_y[t] for t in periods), name="Production capacity at plant")
    m.addConstrs((var_q[i,t]   <= min(inv_cap[i],Q,sum(daily_demand[i,l] for l in periods[t-1:]))*var_z[i,t] for i in customers for t in periods), name="Min quantity delivered only if customer is visited in same period")
    m.addConstrs((quicksum(var_x[i,j,t] for j in nodes if j!=i) == var_z[i,t] for i in customers for t in periods), name="customer visit link")
    m.addConstrs((quicksum(var_x[i,j,t] for j in nodes if j!=i)+quicksum(var_x[j,i,t] for j in nodes if j!=i) == 2*var_z[i,t] for i in customers for t in periods), name="degree constraints at client")
    
    m.addConstrs((quicksum(var_x[0,j,t] for j in customers)== quicksum(var_x[j,0,t] for j in customers)  for t in periods), name="degree constraints at plant client")
    
    m.addConstrs((Q*quicksum(var_x[0,j,t] for j in customers) >= quicksum(var_q[i,t] for i in customers) for t in periods), name="min capacity inequality")


    #m.addConstrs((var_z[0,t,k]   <= var_z[0,t,k-1] for t in periods for k in vehicles[1:]), name="vehicle_symmetry1")
    
    #m.addConstrs((quicksum(var_z[i,t,k] for k in vehicles if k>i)   == 0 for t in periods for i in customers), name="vehicle_symmetryOrderCustomer")

    

    #m.addConstrs((quicksum(var_x[0,j,t,k] for j in customers)  <= quicksum(var_x[0,j,t,k-1] for j in customers)  for t in periods for k in vehicles[1:]), name="vehicle_symmetry")


    #m.addConstrs((quicksum(j*var_x[0,j,t,k] for j in customers)  <= quicksum(j*var_x[j,0,t,k] for j in customers)  for t in periods for k in vehicles), name="symmetric_cost_symmetries")

    
    m.addConstrs((var_x[i,j,t]+var_x[j,i,t]  <= var_z[i,t] for i in customers for j in customers if j!=i for t in periods), name="subtours of size 2")

    
    #initialize initial inventory
    m.addConstrs((var_I[i,0]==initial_inv[i] for i in nodes), name="initial inventory")
    
    #if SINGLE_CUT==False:
    if 0 in probabilities:
        m.addConstr(probabilities[0] * (
            quicksum(r[i] * var_z[i, t] for i in customers for t in periods) - 
            quicksum(c[i, j] * var_x[i, j, t] for i in nodes for j in nodes if i != j for t in periods)
        ) <= var_theta_zero, name=f"Theta0")
    
    #Valid in equalities
    """
    m.addConstr(quicksum(var_y[t] for t in range(1,t1+1))>= 1)
    m.addConstr(quicksum(var_z[0,t,k] for k in vehicles for t in range(1,t2+1))>= float(np.ceil(kt/Q)))
    
    m.addConstrs((var_I[i,t-s-1] >= quicksum(daily_demand[i,t-j] for j in range(0,s+1))*(1-quicksum(var_z[i,t-j,k] for k in vehicles for j in range(0,s+1))) for i in customers for t in periods for s in range(0,t)), name="Inventory inequality")

    m.addConstrs((var_z[i,t,k] <=var_z[0,t,k] for i in customers for t in periods for k in vehicles ), name="routing inequality")
    m.addConstrs((var_x[i,j,t,k] <=var_z[i,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles ), name="routing inequality x")
    m.addConstrs((var_x[i,j,t,k] <=var_z[j,t,k] for i in customers for j in customers if j!=i for t in periods for k in vehicles ), name="routing inequality x")
    """
    if modelCut==False:
        m.addConstrs((quicksum(var_f[j,i,t] for j in nodes if j!=i)-quicksum(var_f[i,j,t] for j in nodes if j!=i)  == var_q[i,t]  for i in customers for t in periods), name="Capacity Flow")
        
        
        m.addConstrs((quicksum(var_f[j,i,t] for j in nodes if j!=i)-quicksum(var_f[i,j,t] for j in nodes if j!=i)  == -quicksum(var_q[l,t] for l in customers) for i in [0] for t in periods), name="Capacity Flow Depot")
    
    
    
        #m.addConstrs((quicksum(var_f[i,j,t] for j in nodes if j!=i)  <= ((Q*var_z[i,t,k])-var_q[i,t,k]) for i in customers for j in nodes if j!=i for t in periods for k in homo_k), name="Capacity Flow Max")
    
        m.addConstrs((var_f[i,j,t]  <= Q*var_x[i,j,t]  for i in nodes for j in nodes if i!=j for t in periods), name="Capacity Flow Max Depot")
    """
    for t,route in [(1,[(0, 8), (8, 0)]),
    (1, [(0, 10), (10, 9), (9, 0)]),
    (2, [(0, 5), (5, 0)]),
    (2, [(0, 7), (7, 6), (6, 0)]),
    (3, [(0, 3), (3, 0)]),
    (3, [(0, 4), (4, 2), (2, 1), (1, 0)]),
    (3, [(0, 9), (9, 0)])]:
        for i,j in route:
            m.addConstr(var_x[i,j,t]==1)"""
    
    m._x = var_x
    m._z = var_z
    m._q = var_q
    m._p = var_p
    m._y = var_y
    m._I = var_I
    m._theta =  var_theta
    if 0 in probabilities:
        m._theta_zero = var_theta_zero
    m._Q = Q
    m._dist = dist
    m._c = dist
    m._r = r
    m._n = num_nodes
    m._nodes = nodes
    m._num_nodes = num_nodes
    m._customers = customers
    m._periods = periods
    m._probabilities = probabilities
    m._benders_cuts = []
    m._theta_scale = {(t,s):None for s in probabilities for t in periods}
    m._found_sol = False
    m._fixed_production_costs = fixed_production_costs
    m._variable_production_costs = variable_production_costs
    m._inventory_holding_costs = inventory_holding_costs
    m._routing_costs = routing_costs
    m._benders_cost = benders_cost
    m._core_point = None
    #m = debug_benders_cut(m)
    
    #m.write("model.lp")
    m.params.TimeLimit = 3600
    m.params.Threads = 16
    #m.params.PreCrush = 1
    m._cb_times = {"l_st_time":[],"u_st_time":[],"l_fcc_time":[],"u_fcc_time":[],"l_benders":[]}
    m._added_cuts = {"lfcc":set(),"ufcc":set(),"lsec":set(),"usec":set()}

    #m.params.Presolve = 0
    #m.params.IntFeasTol = 1e-3
    m.params.LazyConstraints = 1
    m.update()

    
    return m

def reset_submodel_state(submodel):
    """Reset the submodel to its original state"""
    # Remove any constraints added during pareto solving
    constrs_to_remove = [c for c in submodel.getConstrs() if c.ConstrName == "maintain_optimal_obj"]
    for constr in constrs_to_remove:
        submodel.remove(constr)
    submodel.update()
    
def solve_pareto_subproblem_single_cut(m,dObj, core_point,x_dict,z_dict):
    # Constraint to maintain the optimal objective value
    expr_obj = quicksum(m._probabilities[s]*(quicksum(m._r[i]*z_dict[i,t] for i in m._customers if (i,t) in z_dict)-
                       quicksum(m._c[i,j]*x_dict[i,j,t] for i in m._nodes for j in m._nodes if i!=j and (i,j,t) in x_dict)) 
                       for t in m._periods for s in m._scenarios)
    for s in m._scenarios:
        expr_obj += -quicksum(m._lambda1[t,s]*s for t in m._periods)
        expr_obj += -quicksum(m._lambda2[i,j,t,s]*x_dict[i,j,t] for t in m._periods 
                            for i in m._nodes for j in m._nodes if i!=j and (i,j,t) in x_dict)
        expr_obj += -quicksum(m._lambda3[i,t,s]*z_dict[i,t] for t in m._periods 
                            for i in m._customers if (i,t) in z_dict)
        expr_obj += -quicksum(m._lambda6[i,j,t,s] for (i,j) in m._dist for t in m._periods if (i,j,t) in x_dict)
        expr_obj += -quicksum(m._lambda7[i,t,s] for i in m._customers for t in m._periods if (i,t) in z_dict)
    
    # Add constraint to ensure we maintain the optimal objective value
    m.addConstr(expr_obj >= dObj - 1e-6, name="maintain_optimal_obj")
    
    # Define the objective function using the core point
    expr_pareto = quicksum(m._probabilities[s]*(quicksum(m._r[i]*core_point['z'].get((i,t), 0) for i in m._customers)-
                         quicksum(m._c[i,j]*core_point['x'].get((i,j,t), 0) for i in m._nodes for j in m._nodes if i!=j)) 
                         for t in m._periods for s in m._scenarios)
    for s in m._scenarios:
        expr_pareto += -quicksum(m._lambda1[t,s]*s for t in m._periods)
        expr_pareto += -quicksum(m._lambda2[i,j,t,s]*core_point['x'].get((i,j,t), 0) for t in m._periods 
                               for i in m._nodes for j in m._nodes if i!=j)
        expr_pareto += -quicksum(m._lambda3[i,t,s]*core_point['z'].get((i,t), 0) for t in m._periods
                               for i in m._customers)
        expr_pareto += -quicksum(m._lambda6[i,j,t,s] for (i,j) in m._dist for t in m._periods if (i,j,t) in x_dict)
        expr_pareto += -quicksum(m._lambda7[i,t,s] for i in m._customers for t in m._periods if (i,t) in z_dict)
    
    m.setObjective(expr_pareto, GRB.MAXIMIZE)
    
    m.optimize()
    
    # Extract solution values if optimal
    if m.Status == 2:
        vlambda1 = {key: m._lambda1[key].X for key in m._lambda1}
        vlambda2 = {key: m._lambda2[key].X for key in m._lambda2}
        vlambda3 = {key: m._lambda3[key].X for key in m._lambda3}
        vlambda4 = {key: m._lambda4[key].X for key in m._lambda4}
        vlambda5 = {key: m._lambda5[key].X for key in m._lambda5}
        vlambda6 = {k: m._lambda6[k].X for k in m._lambda6}
        vlambda7 = {k: m._lambda7[k].X for k in m._lambda7}
        reset_submodel_state(m)
        return dObj, vlambda1, vlambda2, vlambda3, vlambda4, vlambda5, vlambda6, vlambda7
    
    # Handle other statuses
    print("error, pareto model not solved")
    return dObj, m._lambda1, m._lambda2, m._lambda3, m._lambda4, m._lambda5, m._lambda6, m._lambda7

def solve_pareto_subproblem(m,dObj, core_point,x_dict,z_dict):
    # Constraint to maintain the optimal objective value
    expr_obj = quicksum(m._probability*(quicksum(m._r[i]*z_dict[i,t] for i in m._customers if (i,t) in z_dict)-
                       quicksum(m._c[i,j]*x_dict[i,j,t] for i in m._nodes for j in m._nodes if i!=j and (i,j,t) in x_dict)) 
                       for t in m._periods)
    
    expr_obj += -quicksum(m._lambda1[t,m._scenario]*m._scenario for t in m._periods)
    expr_obj += -quicksum(m._lambda2[i,j,t,m._scenario]*x_dict[i,j,t] for t in m._periods 
                        for i in m._nodes for j in m._nodes if i!=j and (i,j,t) in x_dict)
    expr_obj += -quicksum(m._lambda3[i,t,m._scenario]*z_dict[i,t] for t in m._periods 
                        for i in m._customers if (i,t) in z_dict)
    expr_obj += -quicksum(m._lambda6[i,j,t,m._scenario] for (i,j) in m._dist for t in m._periods if (i,j,t) in x_dict)
    expr_obj += -quicksum(m._lambda7[i,t,m._scenario] for i in m._customers for t in m._periods if (i,t) in z_dict)
    
    # Add constraint to ensure we maintain the optimal objective value
    m.addConstr(expr_obj >= dObj - 1e-6, name="maintain_optimal_obj")
    
    # Define the objective function using the core point
    expr_pareto = quicksum(m._probability*(quicksum(m._r[i]*core_point['z'].get((i,t), 0) for i in m._customers)-
                         quicksum(m._c[i,j]*core_point['x'].get((i,j,t), 0) for i in m._nodes for j in m._nodes if i!=j)) 
                         for t in m._periods)
    
    expr_pareto += -quicksum(m._lambda1[t,m._scenario]*m._scenario for t in m._periods)
    expr_pareto += -quicksum(m._lambda2[i,j,t,m._scenario]*core_point['x'].get((i,j,t), 0) for t in m._periods 
                           for i in m._nodes for j in m._nodes if i!=j)
    expr_pareto += -quicksum(m._lambda3[i,t,m._scenario]*core_point['z'].get((i,t), 0) for t in m._periods
                           for i in m._customers)
    expr_pareto += -quicksum(m._lambda6[i,j,t,m._scenario] for (i,j) in m._dist for t in m._periods if (i,j,t) in x_dict)
    expr_pareto += -quicksum(m._lambda7[i,t,m._scenario] for i in m._customers for t in m._periods if (i,t) in z_dict)
    
    m.setObjective(expr_pareto, GRB.MAXIMIZE)
    
    m.optimize()
    
    # Extract solution values if optimal
    if m.Status == 2:
        vlambda1 = {(t,m._scenario): m._lambda1[t,m._scenario].X for t,m._scenario in m._lambda1}
        vlambda2 = {key: m._lambda2[key].X for key in m._lambda2}
        vlambda3 = {key: m._lambda3[key].X for key in m._lambda3}
        vlambda4 = {key: m._lambda4[key].X for key in m._lambda4}
        vlambda5 = {key: m._lambda5[key].X for key in m._lambda5}
        vlambda6 = {k: m._lambda6[k].X for k in m._lambda6}
        vlambda7 = {k: m._lambda7[k].X for k in m._lambda7}
        reset_submodel_state(m)
        return dObj, vlambda1, vlambda2, vlambda3, vlambda4, vlambda5, vlambda6, vlambda7
    
    # Handle other statuses
    print("error, pareto model not solved")
    return dObj, m._lambda1, m._lambda2, m._lambda3, m._lambda4, m._lambda5, m._lambda6, m._lambda7

def generate_pareto_optimal_cut(model,s, x_dict, z_dict,theta_val, core_point=None, single_cut=False):
    """
    Generate a pareto-optimal Benders cut using a core point approach.
    
    Parameters:
        model: The Gurobi model
        x_dict: Dictionary of x variable values
        z_dict: Dictionary of z variable values
        customers: Set of customer nodes
        dist: Dictionary of distances
        periods: Set of time periods
        probabilities: Dictionary of scenario probabilities
        core_point: A core point in the master problem feasible region (if None, one will be generated)
    
    Returns:
        dObj: Objective value of the dual
        lambda1-lambda7: Dual variable values for generating the cut
    """

    # First solve the regular dual subproblem to get the objective value
    if single_cut==False:
        sub_result = solve_dual_subproblem(x_dict, z_dict, model._submodels[s])
        dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = sub_result
        # Now solve the Magnanti-Wong problem to find a pareto-optimal cut
        return solve_pareto_subproblem(model._submodels[s],dObj, core_point,x_dict,z_dict)
    else:
        sub_result = solve_dual_subproblem_single_cut(x_dict, z_dict, model._submodels[-1])
        dObj, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7 = sub_result
        # Now solve the Magnanti-Wong problem to find a pareto-optimal cut
        return solve_pareto_subproblem_single_cut(model._submodels[-1],dObj, core_point,x_dict,z_dict)
        
       


def build_dual_subproblem(customers, nodes, dist, periods,probability,c,r,scenario,var_cost=False,r_var=0,r_fix=0):
    with gp.Env(empty=True) as env:
    #with gp.Env() as env:
        env.setParam('OutputFlag', 0)
        env.start()
        model_name = f'dual_subproblem_scenario_{scenario}'
        m = gp.Model(model_name, env=env)
        
        # Create dual variables just for this scenario
        lambda1 = {}
        lambda2 = {}
        lambda3 = {}
        lambda4 = {}
        lambda5 = {}
        lambda6 = {}
        lambda7 = {}
        
        # For each period, create the dual variables
        for t in periods:
            # Lambda1 variables
            lambda1[t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                          name=f"ld1[{t},{scenario}]")
            
            # Lambda2 variables
            for i, j in dist:
                lambda2[i, j, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                                    name=f"ld2[{i},{j},{t},{scenario}]")
                lambda6[i, j, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"ld6[{i},{j},{t},{scenario}]")

            
            # Lambda3-7 variables
            #if var_cost==False:
            for i in customers:
                lambda3[i, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                                 name=f"ld3[{i},{t},{scenario}]")
                lambda4[i, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                                 name=f"ld4[{i},{t},{scenario}]")
                lambda5[i, t, scenario] = m.addVar(lb=-float('inf'), ub=float('inf'), 
                                                 vtype=GRB.CONTINUOUS, name=f"ld5[{i},{t},{scenario}]")
                lambda7[i, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Cld7[{i},{t},{scenario}]")
                
          

        
        #m.setObjective(expr, GRB.MAXIMIZE)
        
        # Add constraints for each period
        for t in periods:
            # First constraint type
            for i in nodes:
                for j in nodes:
                    if i != j:
                        if i == 0 and j > 0:
                            expr = -lambda1[t, scenario] + lambda5[j, t, scenario] - lambda2[i, j, t, scenario]-lambda6[i,j,t,scenario]
                            if var_cost==False:
                                RHS = probability * c[i, j]
                            else:
                                RHS = probability * (((r_var-1)*c[i,j])+r_fix)
                            
                            m.addConstr(expr  <= RHS)
                        elif i > 0 and j == 0:
                            if var_cost==False:
                                expr = -lambda2[i, j, t, scenario] + lambda4[i, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                RHS = probability * c[i, j]
                            else:
                                expr = -lambda2[i, j, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                RHS = probability * ((r_var-1)*c[i,j])
                            m.addConstr(expr  <= RHS)
                        elif i > 0 and j > 0:
                            if var_cost==False:
                                expr = lambda4[i, t, scenario] + lambda5[j, t, scenario] - lambda2[i, j, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                RHS = probability * c[i, j]
                            else:
                                expr = -lambda2[i, j, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                RHS = probability * ((r_var-1)*c[i,j])
                            m.addConstr(expr  <= RHS)
            if var_cost==False:
                # Second constraint type
                for i in customers:
                    m.addConstr(-lambda3[i, t, scenario] - lambda4[i, t, scenario] -lambda7[i,t,scenario]<= -probability * r[i])
    m._scenario = scenario
    m._periods = periods
    m._customers = customers
    m._nodes = nodes
    m._dist = dist
    m._c = dist
    m._r_fix = r_fix
    m._r_var = r_var
    m._probability = probability
    m._r = r
    m._lambda1 = lambda1
    m._lambda2 = lambda2
    m._lambda3 = lambda3
    m._lambda4 = lambda4
    m._lambda5 = lambda5
    m._lambda6 = lambda6
    m._lambda7 = lambda7
    m.params.Threads = 4
    return m

def build_dual_subproblem_single_cut(customers, nodes, dist, periods, probabilities,r,var_cost=False,r_var=0,r_fix=0):
    with gp.Env(empty=True) as env:
    #with gp.Env() as env:
        c = dist
        env.setParam('OutputFlag', 0)
        env.start()
        model_name = f'dual_subproblem_single'
        m = gp.Model(model_name, env=env)
        
        # Create dual variables just for this scenario
        lambda1 = {}
        lambda2 = {}
        lambda3 = {}
        lambda4 = {}
        lambda5 = {}
        lambda6 = {}
        lambda7 = {}
        
        for scenario in probabilities:
        
            # For each period, create the dual variables
            for t in periods:
                # Lambda1 variables
                lambda1[t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                              name=f"ld1[{t},{scenario}]")
                
                # Lambda2 variables
                for i, j in dist:
                    lambda2[i, j, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                                        name=f"ld2[{i},{j},{t},{scenario}]")
                    lambda6[i, j, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"ld6[{i},{j},{t},{scenario}]")
    
                
                # Lambda3-7 variables
                #if var_cost==False:
                for i in customers:
                    lambda3[i, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                                     name=f"ld3[{i},{t},{scenario}]")
                    lambda4[i, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                                     name=f"ld4[{i},{t},{scenario}]")
                    lambda5[i, t, scenario] = m.addVar(lb=-float('inf'), ub=float('inf'), 
                                                     vtype=GRB.CONTINUOUS, name=f"ld5[{i},{t},{scenario}]")
                    lambda7[i, t, scenario] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Cld7[{i},{t},{scenario}]")
                    
              
    
            
            #m.setObjective(expr, GRB.MAXIMIZE)
            
            # Add constraints for each period
            for t in periods:
                # First constraint type
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            if i == 0 and j > 0:
                                expr = -lambda1[t, scenario] + lambda5[j, t, scenario] - lambda2[i, j, t, scenario]-lambda6[i,j,t,scenario]
                                if var_cost==False:
                                    RHS = probabilities[scenario] * c[i, j]
                                else:
                                    RHS = probabilities[scenario] * (((r_var-1)*c[i,j])+r_fix)
                                
                                m.addConstr(expr  <= RHS)
                            elif i > 0 and j == 0:
                                if var_cost==False:
                                    expr = -lambda2[i, j, t, scenario] + lambda4[i, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                    RHS = probabilities[scenario] * c[i, j]
                                else:
                                    expr = -lambda2[i, j, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                    RHS = probabilities[scenario] * ((r_var-1)*c[i,j])
                                m.addConstr(expr  <= RHS)
                            elif i > 0 and j > 0:
                                if var_cost==False:
                                    expr = lambda4[i, t, scenario] + lambda5[j, t, scenario] - lambda2[i, j, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                    RHS = probabilities[scenario] * c[i, j]
                                else:
                                    expr = -lambda2[i, j, t, scenario] - lambda5[i, t, scenario]-lambda6[i,j,t,scenario]
                                    RHS = probabilities[scenario] * ((r_var-1)*c[i,j])
                                m.addConstr(expr  <= RHS)
                if var_cost==False:
                    # Second constraint type
                    for i in customers:
                        m.addConstr(-lambda3[i, t, scenario] - lambda4[i, t, scenario] -lambda7[i,t,scenario]<= -probabilities[scenario] * r[i])
    #m._scenario = scenario
    m._periods = periods
    m._customers = customers
    m._nodes = nodes
    m._dist = dist
    m._c = dist
    m._r_fix = r_fix
    m._r_var = r_var
    m._probabilities = probabilities
    m._scenarios = list(probabilities.keys())
    m._r = r
    m._lambda1 = lambda1
    m._lambda2 = lambda2
    m._lambda3 = lambda3
    m._lambda4 = lambda4
    m._lambda5 = lambda5
    m._lambda6 = lambda6
    m._lambda7 = lambda7
    m.params.Threads = 4
    return m

def solve_dual_subproblem_single_cut(x,z,subproblem):
    periods = subproblem._periods
    customers = subproblem._customers
    nodes = subproblem._nodes
    dist = subproblem._dist
    r = subproblem._r
    c = subproblem._dist
    expr = quicksum(subproblem._probabilities[s] * (
        quicksum(r[i] * z[i, t] for i in customers for t in periods if (i, t) in z) - 
        quicksum(c[i, j] * x[i, j, t] for i in nodes for j in nodes if i != j 
               for t in periods if (i, j, t) in x)
    ) for s in subproblem._probabilities)
    
    # Add the dual terms
    for scenario in subproblem._probabilities:
        expr += -quicksum(subproblem._lambda1[t, scenario] * scenario for t in periods)
        expr += -quicksum(subproblem._lambda2[i, j, t, scenario] * x[i, j, t] 
                         for t in periods for i in nodes for j in nodes if i != j and (i, j, t) in x)
        expr += -quicksum(subproblem._lambda3[i, t, scenario] * z[i, t] 
                         for t in periods for i in customers if (i, t) in z)
        
        expr += -quicksum(subproblem._lambda6[i,j,t,scenario] for (i,j) in dist for t in periods)
        expr += -quicksum(subproblem._lambda7[i,t,scenario] for i in customers for t in periods)

    
    subproblem.setObjective(expr, GRB.MAXIMIZE)
    # Optimize
    subproblem.optimize()
    
    # Extract and return solution
    if subproblem.Status == GRB.OPTIMAL:
        obj_val = subproblem.ObjVal
        vlambda1 = {k: subproblem._lambda1[k].X for k in subproblem._lambda1}
        vlambda2 = {k: subproblem._lambda2[k].X for k in subproblem._lambda2}
        vlambda3 = {k: subproblem._lambda3[k].X for k in subproblem._lambda3}
        vlambda4 = {k: subproblem._lambda4[k].X for k in subproblem._lambda4}
        vlambda5 = {k: subproblem._lambda5[k].X for k in subproblem._lambda5}
        vlambda6 = {k: subproblem._lambda6[k].X for k in subproblem._lambda6}
        vlambda7 = {k: subproblem._lambda7[k].X for k in subproblem._lambda7}
        return [obj_val, vlambda1, vlambda2, vlambda3, vlambda4, vlambda5,vlambda6,vlambda7]
    
    # Return None values if not optimal
    return [None, {}, {}, {}, {}, {}, {}, {},{}, {}]

def solve_dual_subproblem(x,z,subproblem):
    scenario = subproblem._scenario 
    periods = subproblem._periods
    customers = subproblem._customers
    nodes = subproblem._nodes
    dist = subproblem._dist
    probability = subproblem._probability
    r = subproblem._r
    c = subproblem._dist
    expr = probability * (
        quicksum(r[i] * z[i, t] for i in customers for t in periods if (i, t) in z) - 
        quicksum(c[i, j] * x[i, j, t] for i in nodes for j in nodes if i != j 
               for t in periods if (i, j, t) in x)
    )
    
    # Add the dual terms
    expr += -quicksum(subproblem._lambda1[t, scenario] * scenario for t in periods)
    expr += -quicksum(subproblem._lambda2[i, j, t, scenario] * x[i, j, t] 
                     for t in periods for i in nodes for j in nodes if i != j and (i, j, t) in x)
    expr += -quicksum(subproblem._lambda3[i, t, scenario] * z[i, t] 
                     for t in periods for i in customers if (i, t) in z)
    
    expr += -quicksum(subproblem._lambda6[i,j,t,scenario] for (i,j) in dist for t in periods)
    expr += -quicksum(subproblem._lambda7[i,t,scenario] for i in customers for t in periods)

    
    subproblem.setObjective(expr, GRB.MAXIMIZE)
    # Optimize
    subproblem.optimize()
    
    # Extract and return solution
    if subproblem.Status == GRB.OPTIMAL:
        obj_val = subproblem.ObjVal
        vlambda1 = {k: subproblem._lambda1[k].X for k in subproblem._lambda1}
        vlambda2 = {k: subproblem._lambda2[k].X for k in subproblem._lambda2}
        vlambda3 = {k: subproblem._lambda3[k].X for k in subproblem._lambda3}
        vlambda4 = {k: subproblem._lambda4[k].X for k in subproblem._lambda4}
        vlambda5 = {k: subproblem._lambda5[k].X for k in subproblem._lambda5}
        vlambda6 = {k: subproblem._lambda6[k].X for k in subproblem._lambda6}
        vlambda7 = {k: subproblem._lambda7[k].X for k in subproblem._lambda7}
        return [obj_val, vlambda1, vlambda2, vlambda3, vlambda4, vlambda5,vlambda6,vlambda7]
    
    # Return None values if not optimal
    return [None, {}, {}, {}, {}, {}, {}, {},{}, {}]


def fix_routes_in_master(model,fixed_routes):
    counter = 0
    fixed_arcs = []
    for t,k in fixed_routes:
        route = fixed_routes[(t,k)]
        print(f"Fix route in period {t} with sequence:{route['sequence']}")
        i = int(route["sequence"][0])
        for j in route["sequence"][1:]:
            j = int(j)
            model.addConstr(model._x[i,j,t]==1.0)
            if j in model._customers:
                model.addConstr(model._z[j,t]==1.0)
            fixed_arcs.append((i,j,t))
            i = j
        counter +=1
    #"""
    for i in model._nodes:
        for j in model._nodes:
            if j!=i:
                for t in model._periods:
                    if (i,j,t) not in fixed_arcs:
                        model.addConstr(model._x[i,j,t]==0)#"""
    print(f"Fixed {counter} routes!")
    return model

def build_master_problem_for_benders_enhanced(dataFile, recourse_gamma,stoch=True, fixed_routes=None):
    """
    Enhanced version of your existing function with solution retention tracking
    """
    # Your existing model building code...
    m = build_master_problem_for_benders(dataFile,recourse_gamma, stoch)
    if fixed_routes!=None:
        m = fix_routes_in_master(m, fixed_routes)
    # Add missing attributes for the callback tracking
    m._cb_last_lower_bound = -float('inf')
    m._cb_last_ub = float('inf') 
    m._cb_lastnode = 0
    
    # Add solution retention parameters
    m.params.PoolSolutions = 10   # Keep multiple good solutions
    #m.params.PoolGap = 0.1       # Keep solutions within 10% of best
    
    # Initialize best solution tracking
    m._best_feasible_obj = float('inf')
    
    # Initialize pending solution storage
    m._pending_solution = None
    m._has_pending_solution = False
    # Initialize solution pool
    m._solution_pool = SolutionPool(max_size=10)
    #m.params.FeasibilityTol = 1e-4
    
    return m

def solve_benders_bc(file_path,recourse_gamma,fixed_routes=None):
    # Parse instance and build model (using your existing code)
    dataFile = parse_mvprp_instance(file_path)
    
    # Build the master problem model (your existing model building code)
    #m = build_master_problem_for_benders(dataFile)  # You'll need to extract this from your main code
    
    # Build the master problem model with enhancements
    m = build_master_problem_for_benders_enhanced(dataFile,recourse_gamma,fixed_routes=fixed_routes)
    #m = debug_benders_cut(m)
    submodels = {}
    if SINGLE_CUT:
        submodels[-1] = build_dual_subproblem_single_cut(m._customers, m._nodes, m._dist, m._periods, m._probabilities, m._r,)
    else:
        for s in m._probabilities:
            submodels[s] = build_dual_subproblem(m._customers, m._nodes, m._dist, m._periods, m._probabilities[s], m._dist, m._r, s)
    m._submodels = submodels
    construction_runtime = 0.0
    if MIPstart==True and fixed_routes==None:
        m.params.StartNumber = 0
        construction_success, construction_runtime = create_main_model_with_mipstart_model1(
            file_path, m, recourse_gamma=3, stoch=stoch
        )
        m.params.TimeLimit = 3600-construction_runtime
        if construction_success:
            print("Construction procedure MIP start applied successfully!")
        else:
            print("Construction procedure failed, proceeding without MIP start")
    if calc_root==True:
         m.setParam('NodeLimit', 1)
    m.optimize(callback=cb)
    
    used_vehicles = set()
    cost_values = []
    costs_vars = [m._fixed_production_costs, m._variable_production_costs, m._inventory_holding_costs, m._routing_costs, 0, 0, m._benders_cost]
    error = ""
    if m.status==3:
       m.computeIIS()
       m.write("infeasible_model.ilp")
    
    if m.status == 2 or m.SolCount > 0:
        for iter2 in range(1):
                m.setParam(GRB.Param.SolutionNumber, iter2)
                print('%g ' % m.PoolObjVal, end='\n')
                for v in m.getVars():
                     if v.xn > 1e-5 or v.xn <0:
                           #print ('%s %g' % (v.varName, v.xn))
                           print ('%s %g' % (v.varName, v.xn))
                print("\n")
        print("\n")
        
        for var in costs_vars:
            if isinstance(var, gp.LinExpr):
                cost_values.append(var.getValue())
                print(var.getValue())
            else:
                print(var)
                cost_values.append(var)
        
        for t in m._periods:
            x_arcs = [(e[0], e[1]) for e in m._x if (m._x[e].x > 0.5 and e[2] == t)]
            if len(x_arcs) > 0:
                q_vals = {e[0]: m._q[e].x for e in m._q if e[1] == t}
                tours, error = check_tour_capacity(x_arcs, q_vals, t, m._Q, error)
    else:
        for var in costs_vars:
            cost_values.append(-1)
    obj = None
    try:
        obj = m.ObjVal
    except:
        obj = None
    cost_labels = ["fixed_prod_cost","variable_prod_cost","inv_hold_cost","routing_cost","recourse_cost","unused_routing_cost","expected_cost_diff"]
    results = [file_path.split("/")[-1],m.ModelName,gurobi_status_dict[m.status],obj,m.ObjBound, m.MIPGap, m.Runtime+construction_runtime,sorted(used_vehicles),m._Q,error,recourse_gamma]+cost_values+[len(m._benders_cuts)]
    column_labels = ["instance","Model","status","ObjVal","ObjBound","Gap","runtime","period_vehicle","Q","error","recourse_gamma"]+cost_labels+["benders_cuts"]
    for key in m._added_cuts:
        results.append(len(m._added_cuts[key]))
        column_labels.append(key)
    for key in m._cb_times:
        results.append(np.array(m._cb_times[key]).mean())
        column_labels.append(key)    
    return results, column_labels
    
def run_benders_bc_experiments():
    """
    Run your experiments using classical Benders decomposition
    """
    mainFolderPath = "./Instances/MVPRP_DatasetI1"

    folder = os.fsencode(mainFolderPath)
    filenames = []
    for subdir, dirs, files in os.walk(mainFolderPath):
        for file in files:
            #if "_V2_" in file:
                experiment_folder = subdir.split(os.sep)[-1]
                filepath = os.path.join(subdir, file)
                filenames.append(filepath)
                
    recourse_gamma = 3.0
    #filenames =["./Instances/Data_Test/MVPRP_C10_P3_V3_I1.dat"]
    """
    filenames =["./Instances/MVPRP_DatasetI1/MVPRP_C15_P9_V3_I1.dat",
                "./Instances/MVPRP_DatasetI1/MVPRP_C15_P9_V2_I1.dat",
                "./Instances/MVPRP_DatasetI1/MVPRP_C50_P3_V3_I1.dat",
                "./Instances/MVPRP_DatasetI1/MVPRP_C20_P9_V3_I1.dat"]"""


    #filenames =["./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V4_I1.dat"]
    #filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P6_V4_I1.dat"]
    #filenames = ["./Instances/Data_Test/MVPRP_C30_P6_V4_I1.dat"]
    #filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C20_P9_V2_I1.dat"]


    #filenames = ["./Instances/Data_Test/MVPRP_C10_P9_V2_I1.dat","./Instances/Data_Test/MVPRP_C15_P9_V3_I1.dat"]
    """
        filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C40_P6_V4_I1.dat",
    "./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V3_I1.dat",
    "./Instances/MVPRP_DatasetI1/MVPRP_C25_P9_V3_I1.dat",
    "./Instances/MVPRP_DatasetI1/MVPRP_C40_P6_V3_I1.dat",
    "./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V4_I1.dat"]"""
    filenames =[".\Instances\DATA_MVPRP_Rev\MVPRP_C10_P3_V3_I1.dat"]

    results = []
    for i in range(1):
        for file_path in filenames:
                result, column_labels = solve_benders_bc(file_path,recourse_gamma)
                results.append(result)
                # Save intermediate results
                df = pd.DataFrame(results, columns=column_labels)
                df.to_csv("results/benders_bc_ParetoOpt_DelayedSolInjection_singleSubproblem_Test2.csv", index=False)
                
        
    return results    

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
            
            route_key = (period,vehicle)
            routes[route_key] = {
                'sequence': route_sequence,
                'distance': total_distance,
                'segments': len(route_data),
                'period': period,
                'vehicle': vehicle
            }
    
    return routes

def run_benders_stochastic_value_calc():
    """
    Run your experiments using classical Benders decomposition
    """
    mainFolderPath = "./Instances/MVPRP_DatasetI1"

    folder = os.fsencode(mainFolderPath)
    filenames = []
    for subdir, dirs, files in os.walk(mainFolderPath):
        for file in files:
            #if "_V2_" in file:
                experiment_folder = subdir.split(os.sep)[-1]
                filepath = os.path.join(subdir, file)
                filenames.append(filepath)
    results = []            
    for recourse_gamma in [1.5,2.0,2.5]:
        #recourse_gamma = 1.5
        
        route_info_path = "routes/model1_ScenarioPRP_homogeneousFractionalCapCut_capCuts_determ_medPlus2_integral_MIPstart_20250812"
        
        # File pattern to match
        file_pattern = "*_sol.csv"  # Change this to match your files
        
        # Check if folders exist
        print("Folder Check:")
        exists = "✓" if os.path.exists(route_info_path) else "✗"
        print(f"  {exists} : {route_info_path}")
        # Load all folders at once
    
        route_data = load_folder_data_routes(route_info_path, file_pattern)
    
        
        #filenames =["./Instances/Data_Test/MVPRP_C10_P3_V3_I1.dat"]
        #filenames =[".\Instances\DATA_MVPRP_Rev\MVPRP_C15_P6_V2_I1.dat"]
    
        #filenames =["./Instances/MVPRP_DatasetI1/MVPRP_C15_P9_V3_I1.dat"]
        #filenames =["./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V4_I1.dat"]
        #filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P6_V4_I1.dat"]
        #filenames = ["./Instances/Data_Test/MVPRP_C30_P6_V4_I1.dat"]
        #filenames = ["./Instances/MVPRP_DatasetI1/MVPRP_C30_P9_V4_I1.dat"]
    
    
        #filenames = ["./Instances/Data_Test/MVPRP_C10_P9_V2_I1.dat","./Instances/Data_Test/MVPRP_C15_P9_V3_I1.dat"]
    
        
        for file_path in filenames:
                route_name = file_path.split("/")[-1].strip(".dat")+"_sol"
                route_sequences = analyze_routes(route_data[route_name])
                result, column_labels = solve_benders_bc(file_path,recourse_gamma,fixed_routes=route_sequences)
                results.append(result)
                # Save intermediate results
                df = pd.DataFrame(results, columns=column_labels)
                df.to_csv("results/benders_bc_ValueOfStochasticInfoGAmma_I1.csv", index=False)
                
        
    return results    

def debug_benders_cut(model):
    cut = []
    var_keys = ["x","z","p","y","I"]
    new_vars = []
    new_vals = []
    #cut = model._benders_cuts[-1]
    for key in var_keys:
        for e in cut[key]:
            expr = 0.0
            if key=="x":
                expr = model._x[e]
            elif key=="z":
                expr = model._z[e]
            elif key =="q":
                expr = model._q[e]
            elif key == "p":
                expr = model._p[e]
            elif key =="y":
                expr = model._y[e]
            elif key == "I":
                expr = model._I[e]
            new_vars.append(expr)
            new_vals.append(cut[key][e])
            model.addConstr(expr == cut[key][e])
    #model.cbSetSolution(new_vars, new_vals)
    #model.cbUseSolution()
    return model

df = run_benders_bc_experiments()
#df_val = run_benders_stochastic_value_calc()