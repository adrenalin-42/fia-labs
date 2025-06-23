import copy
import random
import networkx as nx
import matplotlib.pyplot as plt

class CSP:
    """
    A class to represent a Constraint Satisfaction Problem.
    (This class remains unchanged)
    """
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def is_consistent(self, variable, value, assignment):
        for neighbor in self.constraints.get(variable, []):
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

def select_unassigned_variable(variables, assignment, domains):
    """
    Selects the next variable to assign using the Minimum Remaining Values (MRV) heuristic.
    (This function remains unchanged)
    """
    unassigned_vars = [var for var in variables if var not in assignment]
    return min(unassigned_vars, key=lambda var: len(domains[var]))

def backtrack(assignment, csp):
    """
    The core recursive backtracking algorithm with forward checking.
    (This function remains unchanged)
    """
    if len(assignment) == len(csp.variables):
        return assignment

    var = select_unassigned_variable(csp.variables, assignment, csp.domains)
    local_domains = copy.deepcopy(csp.domains)

    for value in local_domains[var]:
        if csp.is_consistent(var, value, assignment):
            assignment[var] = value
            pruned = {neighbor: [] for neighbor in csp.constraints[var]}
            for neighbor in csp.constraints.get(var, []):
                if neighbor not in assignment:
                    if value in csp.domains[neighbor]:
                        csp.domains[neighbor].remove(value)
                        pruned[neighbor].append(value)

            result = backtrack(assignment, csp)
            if result is not None:
                return result

            del assignment[var]
            for neighbor, values in pruned.items():
                csp.domains[neighbor].extend(values)

    return None

def backtracking_search(csp):
    """Initiates the backtracking search."""
    return backtrack({}, csp)

def visualize_schedule(solution, constraints):
    """
    Creates and displays a graph visualization of the scheduled gate network.
    (This function remains unchanged)
    """
    G = nx.Graph()
    color_map = {'Red': '#ff6347', 'Green': '#90ee90', 'Blue': '#87ceeb', 'Yellow': '#fffacd', 'Purple': '#dda0dd'}

    # Add edges based on the constraints, which also adds the nodes
    for node, neighbors in constraints.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Add any nodes that might not have links (isolated gates)
    for node in solution.keys():
        if node not in G.nodes():
            G.add_node(node)

    # Get node colors in the correct order
    node_colors = [color_map.get(solution[node], 'grey') for node in G.nodes()]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    nx.draw(G, pos,
            labels={node: node.split(':')[0] for node in G.nodes()},
            with_labels=True, node_color=node_colors, node_size=3500,
            font_size=11, font_weight='bold', edge_color='gray', width=1.5, alpha=0.9)

    plt.title("Lunar Council's Orbital Gate Network Schedule", size=18)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{slot} Slot',
                              markerfacecolor=color, markersize=15) for slot, color in color_map.items() if slot in solution.values()]
    plt.legend(handles=legend_elements, title='Time Slots', loc='upper right', fontsize='large')
    plt.show()

# --- NEW FUNCTION FOR RANDOM LINKS ---
def generate_random_links(variables, num_links):
    """
    Generates a random set of constraints (links) for the given variables.

    Args:
        variables (list): The list of gates.
        num_links (int): The total number of links to create.

    Returns:
        dict: A dictionary representing the links.
    """
    links = {var: [] for var in variables}
    possible_pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            possible_pairs.append((variables[i], variables[j]))

    # Shuffle and select a number of links to create
    random.shuffle(possible_pairs)

    for i in range(min(num_links, len(possible_pairs))):
        u, v = possible_pairs[i]
        links[u].append(v)
        links[v].append(u)

    return links

# --- Main execution ---
if __name__ == "__main__":
    print("🌙 Launching the Lunar Council's Orbital Gate Scheduler...\n")

    # 1. Define the problem variables
    gates = ['G1: Command Center', 'G2: Hydroponics', 'G3: Engineering', 'G4: Habitation', 'G5: Landing Pad']
    time_slots = ['Red', 'Green', 'Blue']
    gate_domains = {gate: list(time_slots) for gate in gates}

    # 2. Generate random links instead of using a fixed set
    # You can change the number of links to see different results
    num_random_links = 7
    print(f"Generating a random network with {num_random_links} links...")
    links = generate_random_links(gates, num_random_links)

    # 3. Create and solve the CSP
    lunar_csp = CSP(gates, gate_domains, links)
    solution = backtracking_search(lunar_csp)

    # 4. Print the result and visualize it
    if solution:
        print("✅ Conflict-free schedule found!\n")
        print("---------------------------------------")
        print("| Gate                  | Time Slot   |")
        print("---------------------------------------")
        for gate, slot in sorted(solution.items()):
            print(f"| {gate:<21} | {slot:<11} |")
        print("---------------------------------------")

        print("\nGenerating visual representation...")
        visualize_schedule(solution, links)
    else:
        print(f"❌ No conflict-free schedule could be found with {len(time_slots)} time slots for this network configuration.")
        print("    Try running the script again for a different random network, or increase the number of time slots.")
