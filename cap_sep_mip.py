# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:58:53 2025

@author: un_po
"""

import gurobipy as gp
from gurobipy import GRB, quicksum





def separate_fractional_capacity_inequalities(prp_model,x,z,q,Q,n,t,k=None,mipsol=False, callback=False):
    
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            cur_obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if cur_obj >= epsilon_2:
                S = {i for i in s if model.cbGetSolution(s[i]) > 0.5}
                """
                if mipsol==False and frozenset(S) not in prp_model._added_cuts["ufcc"]:
                    S_comp = [i for i in range(0,n) if i not in S]
                    for t in prp_model._periods:
                        if k==None:
                            Constr_LHS = Q*quicksum(prp_model._x[i, j,t] for i in S for j in S_comp if i!=j)
                            #Constr_RHS = quicksum((Q*prp_model._z[i,t])-prp_model._q[i,t] for i in S if i>0)
                            Constr_RHS = quicksum(prp_model._q[i,t] for i in S if i>0)
                            #prp_model.cbCut(Constr_LHS<=Constr_RHS)
                        else:
                            Constr_LHS = Q*quicksum(prp_model._x[i, j,t,k] for i in S for j in S_comp if i!=j)
                            Constr_RHS = quicksum(prp_model._q[i,t,k] for i in S if i>0)
                        prp_model.cbCut(Constr_LHS>=Constr_RHS)
                    prp_model._added_cuts["ufcc"].add(frozenset(S))
                    model._found_cuts += 1
                    #print("added fractional capacity cut",k, S, cur_obj,model.cbGetSolution(alpha) )
                if mipsol==True:"""
                #Constr_LHS = Q*quicksum(prp_model._x[i, j,t] for i in S for j in S if i!=j)
                #Constr_RHS = quicksum((Q*prp_model._z[i,t])-prp_model._q[i,t] for i in S if i>0)
                #prp_model.cbLazy(Constr_LHS<=Constr_RHS)
                S_comp = [i for i in range(0,n) if i not in S]
                for t in prp_model._periods:
                    if k==None:
                        Constr_LHS = Q*quicksum(prp_model._x[i, j,t] for i in S for j in S_comp if i!=j)
                        Constr_RHS = quicksum(prp_model._q[i,t] for i in S if i>0)
                    else:
                        Constr_LHS = Q*quicksum(prp_model._x[i, j,t,k] for i in S for j in S_comp if i!=j)
                        Constr_RHS = quicksum(prp_model._q[i,t,k] for i in S if i>0)
                    prp_model.cbLazy(Constr_LHS>=Constr_RHS)
                prp_model._added_cuts["lfcc"].add(frozenset(S))
                model._found_cuts += 1
                    #print("added fractional lazy capacity cut", S, cur_obj,model.cbGetSolution(alpha) )

    if len(x)<1:
        return
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            h = {}
            s = {}
            epsilon_1 = 0.5
            epsilon_2 = pow(10, -3)
            for i,j in x:
                if i>0 and j >0:
                    h[i,j] = model.addVar(vtype=GRB.BINARY, name=f'h_{i}_{j}')
                    if i not in s:
                        s[i] =  model.addVar(vtype=GRB.BINARY, name=f's_{i}')
                    if j not in s:
                        s[j] =  model.addVar(vtype=GRB.BINARY, name=f's_{j}')
            alpha = model.addVar(vtype=GRB.INTEGER, name='ALPHA')
            
            model.modelSense = GRB.MAXIMIZE
            model.setObjective(quicksum(x[e]["capacity"]*h[e] for e in h)-quicksum(z.get(j,0)*s[j] for j in s)+1+alpha)
            
            model.addConstrs((h[e] <= s[i] for i in s for e in h if e[0]==i), name=f"link edges_{i}")
            model.addConstrs((h[e] <= s[j] for j in s for e in h if e[1]==j), name=f"link edges_{j}")
            model.addConstr(quicksum(s[j] for j in s)>=2)
            model.addConstr(quicksum(q.get(j,0)*s[j] for j in s)>=Q * alpha+epsilon_1)
            
            #model.Params.OutputFlag = 0
            #model.Params.LogToConsole=0
            model.Params.Threads = 4
            model._found_cuts = 0
            if callback==True:
                model.optimize(mycallback)
            else:
                model.optimize()
                if model.Status==2 or model.SolCount>0:
                    if model.ObjVal >epsilon_2:
                        S = {i for i in s if s[i].Xn > 0.5}
                        S_comp = [i for i in range(0,n) if i not in S]
                        if k==None:
                            Constr_LHS = Q*quicksum(prp_model._x[i, j,t] for i in S for j in S_comp if i!=j)
                            Constr_RHS = quicksum(prp_model._q[i,t] for i in S if i>0)
                        else:
                            Constr_LHS = Q*quicksum(prp_model._x[i, j,t,k] for i in S for j in S_comp if i!=j)
                            Constr_RHS = quicksum(prp_model._q[i,t,k] for i in S if i>0)
                        prp_model.cbLazy(Constr_LHS>=Constr_RHS)
                        prp_model._added_cuts["lfcc"].add(frozenset(S))
                        model._found_cuts += 1
            env.close()
            if model._found_cuts>0:
                return True
            return False