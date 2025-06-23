***FCIM.FIA - Fundamentals of Artificial Intelligence***

> **Lab 3:** *Domain reduction and Constraints* \
> **Performed by:** *Dumitru Moraru*, group *FAF-212* \
> **Verified by:** Elena Graur, asist. univ.

Imports and Utils


```python
import copy
import random
import networkx as nx
import matplotlib.pyplot as plt
```

# Core Problem Representation

* **Variables:** Each Orbital Gate is a variable. Let's say we have gates $\{G_1, G_2, G_3, ..., G_n\}$.
* **Domains:** The domain for each gate is the set of all available time slots. Let's call the set of time slots $T = \{TS_1, TS_2, TS_3, ..., TS_m\}$. Initially, every gate has the same domain $T$.
* **Constraints:** The core rule is that if two gates, $G_i$ and $G_j$, are linked, they cannot be assigned the same time slot. Formally, if `IsLinked(Gi, Gj)`, then their assigned time slots must be different: $Time(G_i) \neq Time(G_j)$.

This structure is equivalent to a famous problem called the **Graph Coloring Problem**, where the gates are the vertices of a graph, the links are the edges, and the time slots are the "colors" we need to assign.


```python
class CSP:
    """
    A class to represent a Constraint Satisfaction Problem.
    """
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def is_consistent(self, variable, value, assignment):
        """
        Check if a given assignment is consistent with the constraints.
        """
        for neighbor in self.constraints.get(variable, []):
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True
```

Defines a generic and reusable blueprint for a Constraint Satisfaction Problem. It abstracts the structure of the problem (variables, domains, constraints) away from the algorithm that solves it. This allows the same solver engine to be used for different problems just by changing the input.

# The Solver Engine
It uses a recursive backtracking algorithm enhanced with heuristics to find a solution efficiently.


```python
def select_unassigned_variable(variables, assignment, domains):
    """
    Selects the next variable to assign using the Minimum Remaining Values (MRV) heuristic.
    """
    unassigned_vars = [var for var in variables if var not in assignment]
    return min(unassigned_vars, key=lambda var: len(domains[var]))

def backtrack(assignment, csp):
    """
    The core recursive backtracking algorithm with forward checking.
    """
    if len(assignment) == len(csp.variables):
        return assignment

    var = select_unassigned_variable(csp.variables, assignment, csp.domains)
    local_domains = copy.deepcopy(csp.domains) # Use a copy for iteration

    for value in local_domains[var]:
        if csp.is_consistent(var, value, assignment):
            assignment[var] = value
            # Forward checking: temporarily remove value from neighbors' domains
            pruned = {neighbor: [] for neighbor in csp.constraints[var]}
            for neighbor in csp.constraints.get(var, []):
                if neighbor not in assignment and value in csp.domains[neighbor]:
                    csp.domains[neighbor].remove(value)
                    pruned[neighbor].append(value)
            
            result = backtrack(assignment, csp)
            if result is not None:
                return result

            # Backtrack: undo assignment and restore pruned domains
            del assignment[var]
            for neighbor, values in pruned.items():
                csp.domains[neighbor].extend(values)
    return None

def backtracking_search(csp):
    """Initiates the backtracking search."""
    return backtrack({}, csp)
```

Explanation
-   **`backtracking_search(csp)`**: A simple "starter" function that initializes an empty assignment and calls the main recursive `backtrack` function.
-   **`select_unassigned_variable(...)`**: This helper function implements a crucial heuristic called **Minimum Remaining Values (MRV)**. Instead of picking the next gate randomly, it intelligently selects the _most constrained_ variable—the one with the fewest available time slots left in its domain. This strategy helps to identify dead-ends much faster, significantly pruning the search tree.
-   **`backtrack(assignment, csp)`**: The core recursive function. For a given partial assignment, it performs these steps:
    1.  **Base Case:** If the assignment is complete (all gates have a time slot), a solution has been found, and it is returned.
    2.  **Variable Selection:** It calls `select_unassigned_variable` to pick the next gate to assign.
    3.  **Value Iteration:** It loops through each possible `value` (time slot) in the chosen variable's domain.
    4.  **Forward Checking:** If the value is consistent, it’s tentatively assigned. Then, this value is _temporarily removed_ from the domains of all neighboring (linked) gates. This propagation of constraints is a key optimization.
    5.  **Recursion:** It calls itself (`backtrack`) to solve the rest of the problem with the newly reduced domains.
    6.  **Backtracking:** If the recursive call fails to find a solution (returns `None`), it means the tentative assignment was a dead end. The function then **undoes** the assignment and restores the domains of the neighbors to their previous state, trying the next value in the loop.

# Random Problem Generation
Instead of solving the same static puzzle every time, this function generates a different network of gate links on each run.


```python
def generate_random_links(variables, num_links):
    """
    Generates a random set of constraints (links) for the given variables.
    """
    links = {var: [] for var in variables}
    possible_pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            possible_pairs.append((variables[i], variables[j]))
            
    random.shuffle(possible_pairs)
    
    for i in range(min(num_links, len(possible_pairs))):
        u, v = possible_pairs[i]
        links[u].append(v)
        links[v].append(u)
        
    return links
```

-   The `generate_random_links` function takes the list of gates and a target number of links as input.
-   It first creates a list of all possible unique pairs of gates.
-   It then shuffles this list randomly.
-   Finally, it picks the top `num_links` pairs from the shuffled list and builds the `links` dictionary, ensuring each connection is recorded symmetrically (from A to B and B to A).

# Results Visualization
Responsible for taking the final schedule and presenting it as a clear, easy-to-read graph.


```python
def visualize_schedule(solution, constraints):
    """
    Creates and displays a graph visualization of the scheduled gate network.
    """
    G = nx.Graph()
    color_map = {'Red': '#ff6347', 'Green': '#90ee90', 'Blue': '#87ceeb'}
    
    for node, neighbors in constraints.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    for node in solution.keys():
        if node not in G.nodes(): G.add_node(node)
            
    node_colors = [color_map.get(solution.get(node), 'grey') for node in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=0.9)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels={n: n.split(':')[0] for n in G.nodes()}, with_labels=True, 
            node_color=node_colors, node_size=3500, font_size=11, font_weight='bold', 
            edge_color='gray', width=1.5, alpha=0.9)
    
    plt.title("Lunar Council's Orbital Gate Network Schedule", size=18)
    # ... (code for legend creation)
    plt.show()
```

-   The `visualize_schedule` function uses the `networkx` and `matplotlib` libraries.
-   It creates a `networkx` graph object from the solution and the link constraints.
-   It defines a color map to associate time slot names ('Red', 'Green', etc.) with specific plot colors.
-   It draws the graph, where:
    -   Each gate is a **node**.
    -   Each link is an **edge** connecting two nodes.
    -   The **color** of each node corresponds to its assigned time slot.
-   Finally, it adds a title and a legend to make the plot self-explanatory. This provides instant visual confirmation that the solution is valid (i.e., no two connected nodes have the same color).

# Main Execution Block
Sets the initial parameters and calls the functions from the other parts in the correct sequence to produce a result.


```python
if __name__ == "__main__":
    print("Launching the Lunar Council's Orbital Gate Scheduler...\n")

    # 1. Define Parameters
    gates = ['G1: Command Center', 'G2: Hydroponics', 'G3: Engineering', 'G4: Habitation', 'G5: Landing Pad']
    time_slots = ['Red', 'Green', 'Blue']
    gate_domains = {gate: list(time_slots) for gate in gates}

    # 2. Generate Problem Instance
    num_random_links = 7 
    print(f"Generating a random network with {num_random_links} links...")
    links = generate_random_links(gates, num_random_links)

    # 3. Create CSP Object
    lunar_csp = CSP(gates, gate_domains, links)

    # 4. Solve the Problem
    solution = backtracking_search(lunar_csp)

    # 5. Report Results
    if solution:
        print("Conflict-free schedule found!\n")
        # ... (code for printing table)
        print("\nGenerating visual representation...")
        visualize_schedule(solution, links)
    else:
        print(f"No conflict-free schedule could be found...")
```

1.  **Define Parameters:** It sets up the list of `gates` and available `time_slots`.
2.  **Generate Problem:** It calls `generate_random_links` (Sector 3) to create the network for this specific run.
3.  **Instantiate CSP:** It creates an instance of the `CSP` class (Sector 1) using the gates, domains, and newly generated links.
4.  **Solve:** It calls `backtracking_search` (Sector 2) to find a solution.
5.  **Report Results:** It checks if a solution was returned.
    -   If **successful**, it prints the schedule in a formatted table and then calls `visualize_schedule` (Sector 4) to display the graph.
    -   If **unsuccessful**, it prints a clear message indicating that no solution could be found for that specific random network.

# Conclusions:
This project successfully delivered an automated scheduling system for the Orbital Gate Network. By modeling the challenge as a Constraint Satisfaction Problem and using an intelligent backtracking algorithm, the system efficiently generates a valid, conflict-free schedule.

The final design is highly flexible, capable of solving any network configuration, and presents the solution as an intuitive, color-coded graph for immediate visual verification. Ultimately, this provides the Lunar Council with a robust and reliable tool to ensure the safe and efficient operation of its vital infrastructure, establishing a strong foundation for future logistical optimizations.

# Bibliography:

1) https://en.wikipedia.org/wiki/Constraint_satisfaction_problem
2) https://www.geeksforgeeks.org/introduction-to-backtracking-data-structure-and-algorithm-tutorials/
3) https://en.wikipedia.org/wiki/Graph_coloring
4) https://networkx.org/
5) https://matplotlib.org/
