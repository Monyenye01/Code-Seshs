"""
This is an implementation of MinPartialSerialization ILP as in the paper
Limiting the memory footprint when dynamically scheduling DAGs on shared-memory platforms
Loris Marchala,∗ , Bertrand Simona,b, Frédéric Viviena

We consider an instance of the MinPartialSerialization problem, 
given by a DAG G = (V,E) with weights on the edges, and a memory limit M.
The sequential schedule S respecting the memory limit is not required. 
By convention, for any (i,j) ̸∈ E, we set mi,j = 0.
"""

from pulp import (
    LpVariable,
    LpProblem,
    lpSum,
    LpMinimize,
    LpBinary,
    LpStatus,
    value
)


def min_partial_serialization_solve(edges: list, 
                                    vertices:list,
                                    w:dict,
                                    mem_limit: float, 
                                    target_node,
                                    source_node, 
                                    memory_req:dict):
    """
    We consider an instance of the MinPartialSerialization problem,
    given by a DAG G = (V,E) with weights on the edges, and a memory limit M. 
    The sequential schedule S respecting the memory limit is not required.
    By convention, for any (i,j) ̸∈ E, we set memory_req_ i,j = 0
    """

  
    W = sum(w[i] for i in vertices) # sum of processing times of all nodes
   

    # objective: minimize critcal path of graph
    # define problem
    prob = LpProblem("min_partial_serialization", LpMinimize)
    
    # create decision variables
    
    # equal to 1 if edge (i,j) exists in the associated partial serialization, and to 0 otherwise
    # ∀(i, j) ∈ V2, ei,j ∈ {0,1}
    e = LpVariable.dicts("e", [(i, j) for i in vertices for j in vertices], 0, 1, LpBinary)
    
    # If e_i,j = 1, then f_i,j ≥ m_i,j, and f_i,j is null otherwise
    f = LpVariable.dicts("f", [(i, j) for i in vertices for j in vertices], 0)
    
    # use the variables pi to represent the top-level of each task, that is, 
    # their earliest completion time in a parallel schedule with infinitely many processors
    p = LpVariable.dicts("p", vertices, 0)
    
    # objective: minimizing the top-level of t, which is the critical path of the graph
    prob += p[target_node]
    
    # constraints
    # Binary constraint for edge inclusion
    for (i, j) in [(i, j) for i in vertices for j in vertices]:
        prob += e[i, j] >= 0
        prob += e[i, j] <= 1

    # Ensure all edges in the original graph are included
    for (i, j) in edges:
        prob += e[i, j] == 1

    # Transitive closure constraint to ensure acyclicity
    for (i, j, k) in [(i, j, k) for i in vertices for j in vertices for k in vertices]:
        prob += e[i, k] >= e[i, j] + e[j, k] - 1
        
    # No self-loop constraint
    for i in vertices:
        prob += e[i, i] == 0
        
    # Flow lower bound constraint
    for (i, j) in [(i, j) for i in vertices for j in vertices]:
        prob += f[i, j] >= e[i, j] * memory_req.get((i, j), 0)

    # Flow upper bound constraint
    for (i, j) in [(i, j) for i in vertices for j in vertices]:
        prob += f[i, j] <= e[i, j] * mem_limit

    # Flow conservation constraint
    # Ensures flow conservation at each node
    for j in vertices:
        if j != source_node and j != target_node:
            prob += lpSum(f[i, j] for i in vertices) - lpSum(f[j, k] for k in vertices) == 0

    # Source flow constraint
    # Ensures total flow out of the source node does not exceed the memory limit
    prob += lpSum(f[source_node, j] for j in vertices) <= mem_limit

    # Minimum completion time constraint
    for i in vertices:
        prob += p[i] >= w[i]

    # Linearized completion time constraint
    for (i, j) in [(i, j) for i in vertices for j in vertices]:
        prob += p[j] >= w[j] + p[i] - W * (1 - e[i, j])

    # Solve the problem
    prob.solve()

    # Print the results
    print("Status:", LpStatus[prob.status])
    print("Objective value (critical path length):", value(prob.objective))
    for v in prob.variables():
        print(v.name, "=", v.varValue)


    # Extract included edges based on decision variables
    included_edges = [(i, j) for (i, j) in e if e[i, j].varValue > 0.5]

    # Create adjacency list
    adjacency_list = {vertex: [] for vertex in vertices}
    for (i, j) in included_edges:
        adjacency_list[i].append(j)

    return adjacency_list

def topological_sort(adjacency_list):
    # Kahn's Algorithm for Topological Sorting
    from collections import deque
    
    in_degree = {u: 0 for u in adjacency_list}
    for u in adjacency_list:
        for v in adjacency_list[u]:
            in_degree[v] += 1
            
    queue = deque([u for u in adjacency_list if in_degree[u] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        
        for v in adjacency_list[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) == len(adjacency_list):
        return topo_order
    else:
        raise ValueError("The graph is not a DAG, topological sorting is not possible.")
    
    
if __name__ == "__main__":
    
    # Define the graph
    vertices = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'F'), ('E', 'F')]
    w = {'A': 10, 'B': 20, 'C': 20, 'D': 25, 'E': 40, 'F':15}
    memory_req = {('A', 'B'): 2, ('A', 'C'): 1, ('B', 'D'): 6, ('C', 'E'): 5, ('D', 'F'): 4, ('E', 'F'):3}
    mem_limit = 9
    source_node = 'A'
    target_node = 'F'

    # Solve the problem and get adjacency list
    adjacency_list = min_partial_serialization_solve(edges, vertices, w, mem_limit, target_node, source_node, memory_req)
    
    # Find topological order
    topo_order = topological_sort(adjacency_list)
    
    # Print topological order
    print("Topological Order:", topo_order)
    



