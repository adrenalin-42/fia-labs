import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_random_map(num_districts, max_neighbors=3):
    """Generate a random solvable map as a graph."""
    G = nx.Graph()
    for i in range(num_districts):
        G.add_node(i)
    
    for i in range(num_districts):
        num_edges = random.randint(1, max_neighbors)
        neighbors = random.sample(range(num_districts), num_edges)
        for neighbor in neighbors:
            if i != neighbor:
                G.add_edge(i, neighbor)
    
    return G

def is_valid_color(node, color, coloring, G):
    """Check if a color is valid for a given node."""
    for neighbor in G.neighbors(node):
        if neighbor in coloring and coloring[neighbor] == color:
            return False
    return True

def color_map_backtracking(G, colors, node=0, coloring={}):
    """Backtracking solution to color the map."""
    if node == len(G.nodes):
        return coloring
    
    for color in colors:
        if is_valid_color(node, color, coloring, G):
            coloring[node] = color
            result = color_map_backtracking(G, colors, node + 1, coloring)
            if result:
                return result
            del coloring[node]
    
    return None

def visualize_colored_map(G, coloring):
    """Visualize the graph with colors."""
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[coloring[node] for node in G.nodes], node_size=800, cmap=plt.cm.rainbow)
    plt.show()

# Generate and solve
num_districts = 10
G = generate_random_map(num_districts)
colors = ["red", "blue", "green", "yellow", "purple"]
coloring = color_map_backtracking(G, colors)

if coloring:
    visualize_colored_map(G, coloring)
else:
    print("No valid coloring found!")
