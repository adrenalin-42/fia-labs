***FCIM.FIA - Fundamentals of Artificial Intelligence***

> **Lab 2:** *Searching Algorithms* \
> **Performed by:** *Dumitru Moraru*, group *FAF-212* \
> **Verified by:** Elena Graur, asist. univ.

Imports and Utils


```python
import random
import heapq
```

# Task 1
The maze is generated using Depth-First Search (DFS) with a randomized approach. The grid is initialized with walls ('|'), and DFS carves paths by moving in random directions. The generated maze contains:
1) Start point ('S') at the top-left
2) End point ('E') at the bottom-right


```python
import random

def generate_maze(rows, cols):
    """
    Generate a random perfect maze using Randomized Depth-First Search algorithm.
    
    Args:
        rows (int): Number of rows in the maze
        cols (int): Number of columns in the maze
    
    Returns:
        list: 2D grid representing the maze where:
            '|' = Wall
            '.' = Path
            'S' = Start
            'E' = End
    """
    # Initialize the grid with walls
    maze = [['|' for _ in range(cols)] for _ in range(rows)]
    
    # Randomized Depth-First Search (DFS) for maze generation
    def dfs(x, y):
        """
        Recursive DFS function to carve paths through the maze.
        Marks the current cell as a path and recursively explores unvisited neighbors.
        
        Args:
            x (int): Current row position
            y (int): Current column position
        """
        maze[x][y] = '.'  # Mark current cell as path
        # Define four directions: right, down, left, up (moving 2 steps to jump over walls)
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)  # Randomize direction order for non-predictable mazes
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy  # Calculate new position
            # Check if the new position is within bounds and still a wall
            if 1 <= nx < rows-1 and 1 <= ny < cols-1 and maze[nx][ny] == '|':
                # Remove the wall between current cell and new cell
                maze[x + dx//2][y + dy//2] = '.'
                dfs(nx, ny)  # Recursively continue from the new cell
    
    # Start from top-left corner (with 1-cell buffer from the edge)
    maze[1][1] = '.'
    dfs(1, 1)
    
    # Set start and end points
    maze[1][1] = 'S'  # Start at top-left (with 1-cell buffer)
    maze[rows-2][cols-2] = 'E'  # End at bottom-right (with 1-cell buffer)
    return maze
```

The maze is initialized as a grid filled with walls ('|').
The DFS algorithm starts at (1,1), randomly choosing a direction and carving a passage while ensuring no loops.
The start position ('S') is placed at (1,1), and the exit ('E') at (rows-2, cols-2).

# Task 2
The `heuristic` function used in A* is the Manhattan distance, which estimates the cost from the current position to the end.The function `find_path()` is responsible for solving the maze using either Dijkstra's Algorithm or A*.
1) The start ('S') and end ('E') positions are identified.
2) The priority queue (min-heap) is used to keep track of the lowest-cost path.
3) The algorithm expands the neighboring cells and updates the cost of reaching them.
4) A vs. Dijkstra:
    * Dijkstra uses the actual cost (`new_cost`).
    * A* uses `new_cost + heuristic()`, making it more efficient.
5) The shortest path is reconstructed using the `came_from` dictionary.


```python
import heapq

def heuristic(a, b):
    """
    Calculate the Manhattan distance heuristic between two points.
    Used by A* algorithm to estimate distance to goal.
    
    Args:
        a (tuple): First point (row, col)
        b (tuple): Second point (row, col)
    
    Returns:
        int: Manhattan distance between points
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path(maze, algorithm):
    """
    Find the shortest path through the maze using either Dijkstra's or A* algorithm.
    Modifies the maze in-place to mark the solution path with '*'.
    
    Args:
        maze (list): 2D grid representing the maze
        algorithm (str): "Dijkstra" or "A*" to specify which algorithm to use
    """
    rows, cols = len(maze), len(maze[0])
    start, end = None, None
    
    # Locate start and end positions in the maze
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 'S':
                start = (r, c)
            elif maze[r][c] == 'E':
                end = (r, c)
    
    # Implementation for both Dijkstra & A* Algorithm
    # Uses priority queue (via heapq) to always process the node with lowest cost first
    frontier = [(0, start)]  # (priority, position)
    came_from = {start: None}  # Track path for reconstruction
    cost_so_far = {start: 0}  # Track cost to reach each cell
    
    while frontier:
        # Get the cell with lowest priority from frontier
        current_cost, current = heapq.heappop(frontier)
        
        # If we've reached the end, stop searching
        if current == end:
            break
        
        # Explore all four adjacent cells (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if neighbor is valid (within bounds and not a wall)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] != '|':
                # Calculate new cost to this neighbor
                new_cost = cost_so_far[current] + 1
                
                # If this is a new cell or we found a better path to it
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    
                    # Calculate priority based on algorithm choice:
                    # Dijkstra: just the cost so far
                    # A*: cost so far + heuristic (estimated distance to goal)
                    priority = new_cost if algorithm == "Dijkstra" else new_cost + heuristic(neighbor, end)
                    
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
    
    # Reconstruct and mark the solution path
    current = end
    while current and current != start:
        maze[current[0]][current[1]] = '*'
        current = came_from.get(current, None)
    
    # Re-mark start and end points (they might have been overwritten)
    maze[start[0]][start[1]] = 'S'
    maze[end[0]][end[1]] = 'E'
```

# Task 3
The `main()` function generates a maze and solves it using both algorithms.
A 15x15 maze is generated.
The maze is solved using A*, displaying the shortest path.
The maze is solved using Dijkstra, displaying the path found.
The output shows the maze before and after pathfinding.


```python
def main():
    """
    Main function to run the maze generation and solving.
    """
    # Create a maze of size 15x30
    rows, cols = 15, 30
    maze = generate_maze(rows, cols)
    print("Generated Maze:")
    print_maze(maze)
    
    # Solve the maze using A* algorithm
    print("Solving with A*:")
    astar_maze = [row[:] for row in maze]  # Create a deep copy of the maze
    find_path(astar_maze, "A*")
    print_maze(astar_maze)
    
    # Solve the maze using Dijkstra's algorithm
    print("Solving with Dijkstra:")
    dijkstra_maze = [row[:] for row in maze]  # Create a deep copy of the maze
    find_path(dijkstra_maze, "Dijkstra")
    print_maze(dijkstra_maze)

if __name__ == "__main__":
    main()
```

# Conclusions:
This lab successfully implemented maze generation using Depth-First Search and pathfinding using Dijkstra's Algorithm and A*.

* DFS created a solvable maze with randomized paths.
* Dijkstra and A* found the shortest path from 'S' to 'E'.
* A* performed better because it uses heuristics to guide the search.

# Bibliography:

1) https://www.javatpoint.com/mini-max-algorithm-in-ai
2) https://medium.com/@aaronbrennan.brennan/minimax-algorithm-and-alpha-beta-pruning-646beb01566c
3) https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
4) https://stackoverflow.com/questions/20009796/transposition-tables
5) https://www.codecademy.com/resources/docs/ai/search-algorithms/a-star-search
