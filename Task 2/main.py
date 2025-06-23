import random
import heapq

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

def print_maze(maze):
    """
    Print the maze in a more readable format.
    
    Args:
        maze (list): 2D grid representing the maze
    """
    for row in maze:
        # Use different characters for better visibility
        formatted_row = []
        for cell in row:
            if cell == '|':
                formatted_row.append('██')  # Wall as solid block
            elif cell == '.':
                formatted_row.append('  ')  # Path as empty space
            elif cell == 'S':
                formatted_row.append('S ')  # Start
            elif cell == 'E':
                formatted_row.append('E ')  # End
            elif cell == '*':
                formatted_row.append('· ')  # Path solution
            else:
                formatted_row.append(cell + ' ')
        print("".join(formatted_row))
    print()

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