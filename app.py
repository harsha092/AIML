import tkinter as tk
from tkinter import messagebox, simpledialog
import heapq
import heapq
import copy
import random
import math

class PuzzleNode:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(tuple(map(tuple, self.state)))

    def is_goal(self, goal_state):
        return self.state == goal_state

    def generate_children(self):
        x, y = self.find_blank(self.state)
        possible_moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        children = []
        for move in possible_moves:
            child_state = self.make_move(self.state, (x, y), move)
            if child_state is not None:
                child = PuzzleNode(child_state, parent=self, action=move, cost=self.cost + 1,
                                   heuristic=self.calculate_heuristic(child_state))
                children.append(child)
        return children

    def find_blank(self, state):
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 0:
                    return i, j

    def make_move(self, state, current_pos, new_pos):
        x1, y1 = current_pos
        x2, y2 = new_pos
        if 0 <= x2 < len(state) and 0 <= y2 < len(state[0]):
            new_state = copy.deepcopy(state)
            new_state[x1][y1], new_state[x2][y2] = new_state[x2][y2], new_state[x1][y1]
            return new_state
        else:
            return None

    def calculate_heuristic(self, state):
        # Manhattan distance heuristic
        distance = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                value = state[i][j]
                if value != 0:
                    goal_x, goal_y = divmod(value - 1, 3)
                    distance += abs(i - goal_x) + abs(j - goal_y)
        return distance

def distance(city1, city2):
    """ Calculate the Euclidean distance between two cities. """
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def nearest_neighbor(start, unvisited_cities):
    """ Find the nearest unvisited city to the given city. """
    min_distance = float('inf')
    nearest_city = None
    for city in unvisited_cities:
        dist = distance(start, city)
        if dist < min_distance:
            min_distance = dist
            nearest_city = city
    return nearest_city, min_distance

def tsp_nearest_neighbor(cities):
    """ Solve the Traveling Salesman Problem using the Nearest Neighbor Algorithm. """
    if not cities:
        return [], 0
    current_city = cities[0]
    unvisited_cities = set(cities[1:])
    tour = [current_city]
    
    while unvisited_cities:
        next_city, _ = nearest_neighbor(current_city, unvisited_cities)
        tour.append(next_city)
        unvisited_cities.remove(next_city)
        current_city = next_city
    
    tour.append(tour[0])  # Return to the starting city to complete the tour
    total_distance = sum(distance(tour[i], tour[i+1]) for i in range(len(tour) - 1))
    return tour, total_distance

def best_first_search(initial_state, goal_state):
    start_node = PuzzleNode(initial_state, None, None, 0, 0)
    open_set = [start_node]
    closed_set = set()
    while open_set:
        current_node = heapq.heappop(open_set)
        if current_node.is_goal(goal_state):
            return reconstruct_path(current_node)
        closed_set.add(current_node)
        children = current_node.generate_children()
        for child in children:
            if child not in closed_set and all(child != open_node for open_node in open_set):
                heapq.heappush(open_set, child)
    return None

def reconstruct_path(node):
    path = []
    while node.parent:
        path.insert(0, (node.action, node.state))
        node = node.parent
    return path


# Global function definitions for the Water Jug Problem
def water_jug_a_star(jug1_capacity, jug2_capacity, target_amount):
    start_state = (0, 0)
    open_list = [(0, start_state)]  # (f-value, state)
    closed_set = set()
    parents = {start_state: None}

    while open_list:
        current_cost, current_state = heapq.heappop(open_list)
        if current_state == (target_amount, 0) or current_state == (0, target_amount):
            # Goal reached
            path = []
            while current_state:
                path.append(current_state)
                current_state = parents[current_state]
            return path[::-1]

        closed_set.add(current_state)

        successors = generate_successors(current_state, jug1_capacity, jug2_capacity)

        for successor in successors:
            if successor not in closed_set and successor not in [s[1] for s in open_list]:
                cost = current_cost + 1  # Assuming each step has a cost of 1
                heuristic = calculate_heuristic(successor, target_amount)
                f_value = cost + heuristic
                heapq.heappush(open_list, (f_value, successor))
                parents[successor] = current_state

    return None

def generate_successors(state, jug1_capacity, jug2_capacity):
    jug1, jug2 = state
    successors = []
    # Fill jug 1
    successors.append((jug1_capacity, jug2))
    # Fill jug 2
    successors.append((jug1, jug2_capacity))
    # Empty jug 1
    successors.append((0, jug2))
    # Empty jug 2
    successors.append((jug1, 0))
    # Pour water from jug 1 to jug 2
    pour = min(jug1, jug2_capacity - jug2)
    successors.append((jug1 - pour, jug2 + pour))
    # Pour water from jug 2 to jug 1
    pour = min(jug2, jug1_capacity - jug1)
    successors.append((jug1 + pour, jug2 - pour))
    return successors

def calculate_heuristic(state, target_amount):
    # Basic heuristic: Absolute difference between the total amount in both jugs and the target amount
    return abs(sum(state) - target_amount)

# Definitions for the games
#TICTACTOE
class TicTacToe:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic Tac Toe")
        self.turn = True  # True for X's turn, False for O's
        self.moves = 0
        self.buttons = [[None]*3 for _ in range(3)]
        self.create_widgets()
        self.game_over = False

    def create_widgets(self):
        self.turn_label = tk.Label(self.master, text="X's Turn", font=('normal', 14))
        self.turn_label.grid(row=3, columnspan=3)
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(self.master, text='', font=('normal', 20), height=3, width=6,
                                               command=lambda i=i, j=j: self.click(i, j))
                self.buttons[i][j].grid(row=i, column=j)

    def click(self, i, j):
        if not self.game_over and self.buttons[i][j]['text'] == '':
            self.buttons[i][j]['text'] = 'X' if self.turn else 'O'
            self.turn = not self.turn
            self.moves += 1
            winner = self.check_winner()
            if winner:
                self.end_game(winner)
            elif self.moves == 9:
                self.end_game("Draw")
            else:
                self.update_turn_label()

    def check_winner(self):
        for i in range(3):
            if self.buttons[i][0]['text'] == self.buttons[i][1]['text'] == self.buttons[i][2]['text'] != '':
                self.highlight_winner_cells([(i, 0), (i, 1), (i, 2)])
                return self.buttons[i][0]['text']
            if self.buttons[0][i]['text'] == self.buttons[1][i]['text'] == self.buttons[2][i]['text'] != '':
                self.highlight_winner_cells([(0, i), (1, i), (2, i)])
                return self.buttons[0][i]['text']
        if self.buttons[0][0]['text'] == self.buttons[1][1]['text'] == self.buttons[2][2]['text'] != '':
            self.highlight_winner_cells([(0, 0), (1, 1), (2, 2)])
            return self.buttons[0][0]['text']
        if self.buttons[0][2]['text'] == self.buttons[1][1]['text'] == self.buttons[2][0]['text'] != '':
            self.highlight_winner_cells([(0, 2), (1, 1), (2, 0)])
            return self.buttons[0][2]['text']
        return False

    def highlight_winner_cells(self, cells):
        for cell in cells:
            self.buttons[cell[0]][cell[1]]['bg'] = 'lightgreen'

    def end_game(self, winner):
        self.game_over = True
        message = f"{winner} wins after {self.moves} moves!" if winner in ('X', 'O') else "It's a draw!"
        messagebox.showinfo("Game Over", message)
        self.update_turn_label()

    def update_turn_label(self):
        if self.game_over:
            self.turn_label.config(text="")
        else:
            current_player = "X" if self.turn else "O"
            self.turn_label.config(text=f"{current_player}'s Turn")
#Rat Maze
class RatInAMaze:
    def __init__(self, master):
        self.master = master
        self.master.title("Rat in a Maze")
        self.rows = 5
        self.columns = 5
        self.cell_size = 30
        self.walls = set()
        self.cells = []
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.master, width=self.columns * self.cell_size, height=self.rows * self.cell_size)
        self.canvas.pack()
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                cell = self.canvas.create_rectangle(j * self.cell_size, i * self.cell_size, (j + 1) * self.cell_size, (i + 1) * self.cell_size, fill="green", outline="black")
                row.append(cell)
            self.cells.append(row)
        self.canvas.bind("<Button-1>", self.toggle_wall)
        solve_button = tk.Button(self.master, text="Solve Maze", command=self.solve_maze)
        solve_button.pack()
        clear_button = tk.Button(self.master, text="Clear Walls", command=self.clear_walls)
        clear_button.pack()

    def toggle_wall(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if (x, y) in self.walls:
            self.walls.remove((x, y))
            self.canvas.itemconfig(self.cells[x][y], fill="green")
        else:
            self.walls.add((x, y))
            self.canvas.itemconfig(self.cells[x][y], fill="red")

    def clear_walls(self):
        self.walls.clear()
        for i in range(self.rows):
            for j in range(self.columns):
                self.canvas.itemconfig(self.cells[i][j], fill="green")

    def solve_maze(self):
        maze = self.create_maze()
        start = (0, 0)
        goal = (self.rows - 1, self.columns - 1)
        path = self.find_path(maze, start, goal)
        if path:
            messagebox.showinfo("Path found", f"Path: {path}")
        else:
            messagebox.showinfo("No path found", "No path found")

    def create_maze(self):
        maze = []
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                if (i, j) in self.walls:
                    row.append(1)
                else:
                    row.append(0)
            maze.append(row)
        return maze

    def find_path(self, maze, start, goal):
        def dfs(current, path):
            if current == goal:
                return path + [current]
            x, y = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = x + dx, y + dy
                if (0 <= next_x < self.rows and 0 <= next_y < self.columns and 
                    maze[next_x][next_y] == 0 and (next_x, next_y) not in path):
                    result = dfs((next_x, next_y), path + [current])
                    if result:
                        return result
            return None
        return dfs(start, [])

#Water Jug
class WaterJugSolver(tk.Toplevel):
    
    def __init__(self):
        super().__init__()
        self.title("Water Jug Problem Solver")  # Set the title directly in the constructor
        self.geometry("400x300")
        self.jug1_capacity = simpledialog.askinteger("Capacity", "Enter capacity of Jug 1:")
        self.jug2_capacity = simpledialog.askinteger("Capacity", "Enter capacity of Jug 2:")
        self.target_amount = simpledialog.askinteger("Target", "Enter target amount:")
        self.jug1 = 0
        self.jug2 = 0
        self.create_widgets()

    def create_widgets(self):
        self.label_jug1 = tk.Label(self, text=f'Jug 1: 0/{self.jug1_capacity} L')
        self.label_jug1.pack()

        self.label_jug2 = tk.Label(self, text=f'Jug 2: 0/{self.jug2_capacity} L')
        self.label_jug2.pack()

        tk.Button(self, text="Fill Jug 1", command=lambda: self.update_jugs(self.jug1_capacity, self.jug2)).pack()
        tk.Button(self, text="Fill Jug 2", command=lambda: self.update_jugs(self.jug1, self.jug2_capacity)).pack()
        tk.Button(self, text="Empty Jug 1", command=lambda: self.update_jugs(0, self.jug2)).pack()
        tk.Button(self, text="Empty Jug 2", command=lambda: self.update_jugs(self.jug1, 0)).pack()
        tk.Button(self, text="Pour Jug 1 to Jug 2", command=lambda: self.pour_water(1)).pack()
        tk.Button(self, text="Pour Jug 2 to Jug 1", command=lambda: self.pour_water(2)).pack()
        tk.Button(self, text="Clear", command=self.clear).pack()
        tk.Button(self, text="Solve", command=self.solve).pack()
        

    def update_jugs(self, new_jug1, new_jug2):
        self.jug1 = new_jug1
        self.jug2 = new_jug2
        self.label_jug1.config(text=f'Jug 1: {self.jug1}/{self.jug1_capacity} L')
        self.label_jug2.config(text=f'Jug 2: {self.jug2}/{self.jug2_capacity} L')

    def pour_water(self, from_jug):
        if from_jug == 1:
            pour_amount = min(self.jug1, self.jug2_capacity - self.jug2)
            self.update_jugs(self.jug1 - pour_amount, self.jug2 + pour_amount)
        else:
            pour_amount = min(self.jug2, self.jug1_capacity - self.jug1)
            self.update_jugs(self.jug1 + pour_amount, self.jug2 - pour_amount)

    def clear(self):
        self.jug1 = 0
        self.jug2 = 0
        self.label_jug1.config(text=f'Jug 1: {self.jug1}/{self.jug1_capacity} L')
        self.label_jug2.config(text=f'Jug 2: {self.jug2}/{self.jug2_capacity} L')
        
    def solve(self):
        if not self.jug1_capacity or not self.jug2_capacity or not self.target_amount:
            self.jug1_capacity = simpledialog.askinteger("Capacity", "Enter capacity of Jug 1:")
            self.jug2_capacity = simpledialog.askinteger("Capacity", "Enter capacity of Jug 2:")
            self.target_amount = simpledialog.askinteger("Target", "Enter target amount:")
            self.update_jugs(0, 0)
            return

        path = water_jug_a_star(self.jug1_capacity, self.jug2_capacity, self.target_amount)
        if path:
            messagebox.showinfo("Solution", " -> ".join(str(step) for step in path))
        else:
            messagebox.showinfo("Solution", "No solution possible.")

def water_jug_a_star(jug1_capacity, jug2_capacity, target):
    # Implementation of A* algorithm to solve water jug problem
    # Returns a list of steps to reach the target amount or None if no solution exists
    # Each step is represented as a tuple (jug1_amount, jug2_amount)

    def heuristic(state):
        # A heuristic function to estimate the distance from the current state to the goal state
        jug1_amount, jug2_amount = state
        return abs(jug1_amount - target) + abs(jug2_amount - target)

    heap = [(0, (0, 0))]
    visited = set()
    parents = {}

    while heap:
        cost, state = heapq.heappop(heap)
        if state in visited:
            continue
        visited.add(state)
        jug1_amount, jug2_amount = state

        if jug1_amount == target or jug2_amount == target:
            # Goal state reached
            path = []
            while state != (0, 0):
                path.append(state)
                state = parents[state]
            path.append((0, 0))
            return path[::-1]

        # Generate all possible next states
        next_states = []
        next_states.append((jug1_capacity, jug2_amount))  # Fill jug 1
        next_states.append((jug1_amount, jug2_capacity))  # Fill jug 2
        next_states.append((0, jug2_amount))  # Empty jug 1
        next_states.append((jug1_amount, 0))  # Empty jug 2
        next_states.append((min(jug1_amount + jug2_amount, jug1_capacity), max(0, jug1_amount + jug2_amount - jug1_capacity)))  # Pour from jug 2 to jug 1
        next_states.append((max(0, jug1_amount + jug2_amount - jug2_capacity), min(jug1_amount + jug2_amount, jug2_capacity)))  # Pour from jug 1 to jug 2

        for next_state in next_states:
            next_cost = cost + 1  # Cost of each step is 1
            if next_state not in visited:
                heapq.heappush(heap, (next_cost + heuristic(next_state), next_state))
                parents[next_state] = state

    return None  # No solution found
#Puzzle
class EightPuzzleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("8-Puzzle Solver")
        self.initial_state = None
        self.goal_state = None
        self.current_state = None
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_input_widgets()

    def create_input_widgets(self):
        self.entry_widgets_initial = []
        tk.Label(self.master, text="Enter initial state (use 0 for empty space):").grid(row=0, columnspan=3)
        for i in range(3):
            for j in range(3):
                entry = tk.Entry(self.master, width=5)
                entry.grid(row=i+1, column=j)
                self.entry_widgets_initial.append(entry)
        
        self.entry_widgets_goal = []
        tk.Label(self.master, text="Enter goal state (use 0 for empty space):").grid(row=4, columnspan=3)
        for i in range(3):
            for j in range(3):
                entry = tk.Entry(self.master, width=5)
                entry.grid(row=i+5, column=j)
                self.entry_widgets_goal.append(entry)

        tk.Button(self.master, text="Submit", command=self.read_input).grid(row=8, columnspan=3)

    def read_input(self):
        self.initial_state = [[int(self.entry_widgets_initial[i*3 + j].get()) for j in range(3)] for i in range(3)]
        self.goal_state = [[int(self.entry_widgets_goal[i*3 + j].get()) for j in range(3)] for i in range(3)]
        self.current_state = copy.deepcopy(self.initial_state)
        self.create_widgets()
        self.update_buttons()

    def create_widgets(self):
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.master, text='', width=10, height=4,
                                command=lambda x=i, y=j: self.move(x, y))
                btn.grid(row=i+9, column=j)
                self.buttons[i][j] = btn
        tk.Button(self.master, text="Solve", command=self.solve).grid(row=12, column=1)

    def update_buttons(self):
        for i in range(3):
            for j in range(3):
                tile = self.current_state[i][j]
                if tile == 0:
                    self.buttons[i][j].config(text='', bg='SystemButtonFace')
                else:
                    self.buttons[i][j].config(text=str(tile), bg='SystemButtonFace')

    def move(self, x, y):
        blank_x, blank_y = None, None
        for i in range(3):
            for j in range(3):
                if self.current_state[i][j] == 0:
                    blank_x, blank_y = i, j
        # Check if the move is valid
        if (abs(blank_x - x) == 1 and blank_y == y) or (abs(blank_y - y) == 1 and blank_x == x):
            self.current_state[blank_x][blank_y], self.current_state[x][y] = self.current_state[x][y], self.current_state[blank_x][blank_y]
            self.update_buttons()

    def solve(self):
        solution_path = best_first_search(self.current_state, self.goal_state)
        if solution_path:
            self.show_solution(solution_path)
        else:
            messagebox.showinfo("No Solution", "No solution found.")

    def show_solution(self, path):
        self.move_count = len(path) - 1  
        for move, state in path:
            self.current_state = state
            self.update_buttons()
            self.master.update()
            self.master.after(500)
        self.display_move_count()
        
    def display_move_count(self):
        messagebox.showinfo("Solution", f"Solution found in {self.move_count} moves.")

#TravellingSalesman
class TSPGui:
    def __init__(self, master):
        self.master = master
        self.master.title("TSP Nearest Neighbor")
        self.canvas = tk.Canvas(self.master, width=400, height=400, bg='white')
        self.canvas.pack(side=tk.LEFT)
        self.cities = []
        self.city_size = 5
        
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.add_city_btn = tk.Button(control_frame, text="Add Random Cities", command=self.add_random_cities)
        self.add_city_btn.pack(fill=tk.X)
        
        self.manual_input_btn = tk.Button(control_frame, text="Manual City Input", command=self.manual_city_input)
        self.manual_input_btn.pack(fill=tk.X)
        
        self.solve_btn = tk.Button(control_frame, text="Solve TSP", command=self.solve_tsp)
        self.solve_btn.pack(fill=tk.X)
        self.solve_btn.config(state=tk.DISABLED)  # Initially disabled
        
        self.clear_btn = tk.Button(control_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(fill=tk.X)
        self.clear_btn.config(state=tk.DISABLED)  # Initially disabled
        
        self.city_size_scale = tk.Scale(control_frame, label="City Size", from_=1, to=10, orient=tk.HORIZONTAL,
                                        command=self.update_city_size)
        self.city_size_scale.set(self.city_size)
        self.city_size_scale.pack(fill=tk.X)
        
        self.canvas.bind("<Button-1>", self.add_city)

    def add_city(self, event):
        x, y = event.x, event.y
        city_number = len(self.cities) + 1
        self.canvas.create_text(x, y, text=str(city_number), fill='black')
        self.canvas.create_text(x, y + 15, text=f"({x}, {y})", fill='gray')  # Display coordinates
        self.cities.append((x, y))
        self.update_buttons_state()  # Enable solve and clear buttons
        self.draw_tour([])  # Clear previous tour
        messagebox.showinfo("City Added", "City added successfully.")
        
    def add_random_cities(self):
        num_cities = simpledialog.askinteger("Random Cities", "Enter the number of random cities to add:", initialvalue=5)
        if num_cities is not None:
            for _ in range(num_cities):
                x, y = random.randint(20, 380), random.randint(20, 380)
                self.canvas.create_oval(x - self.city_size, y - self.city_size,
                                        x + self.city_size, y + self.city_size, fill='black')
                self.cities.append((x, y))
            self.update_buttons_state()  # Enable solve and clear buttons
            self.draw_tour([])  # Clear previous tour
            messagebox.showinfo("Random Cities Added", f"{num_cities} random cities added successfully.")

    def manual_city_input(self):
        x = simpledialog.askinteger("Manual Input", "Enter the x-coordinate of the city:")
        y = simpledialog.askinteger("Manual Input", "Enter the y-coordinate of the city:")
        if x is not None and y is not None:
            self.canvas.create_oval(x - self.city_size, y - self.city_size,
                                    x + self.city_size, y + self.city_size, fill='black')
            self.cities.append((x, y))
            self.update_buttons_state()  # Enable solve and clear buttons
            self.draw_tour([])  # Clear previous tour
            messagebox.showinfo("Manual City Input", "City added successfully.")

    def solve_tsp(self):
        if not self.cities:
            messagebox.showwarning("TSP Solver", "Please add some cities first.")
            return
        
        tour, total_distance = tsp_nearest_neighbor(self.cities)
        self.draw_tour(tour)
        messagebox.showinfo("TSP Solver", f"TSP solved successfully.\nTotal Distance: {total_distance:.2f}")

    def draw_tour(self, tour):
        self.canvas.delete("line")  # Remove old lines
        if len(tour) > 1:
            for i in range(len(tour) - 1):
                x1, y1 = tour[i]
                x2, y2 = tour[i + 1]
                self.canvas.create_line(x1, y1, x2, y2, fill='red', tags="line")

    def clear_canvas(self):
        self.canvas.delete(tk.ALL)
        self.cities = []
        self.solve_btn.config(state=tk.DISABLED)  # Disable solve button
        self.clear_btn.config(state=tk.DISABLED)  # Disable clear button
        messagebox.showinfo("Canvas Cleared", "Canvas cleared successfully.")

    def update_city_size(self, size):
        self.city_size = int(size)

    def update_buttons_state(self):
        # Enable solve and clear buttons if there are cities
        if self.cities:
            self.solve_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
def main():
    root = tk.Tk()
    root.title("Game Selector")
    root.geometry("300x400")
    icon_path = "C:\\Users\\rhars\\OneDrive\\Desktop\\projects\\aiml\\aiml lab code\\32officeicons-20_89419.ico"
    root.wm_iconbitmap(icon_path)
    
    root.configure(bg="lightblue")
    
    def open_tic_tac_toe():
        ttt_window = tk.Toplevel(root)
        TicTacToe(ttt_window)

    def open_rat_in_a_maze():
        rim_window = tk.Toplevel(root)
        RatInAMaze(rim_window)

    def open_water_jug_solver():
        wj_window = WaterJugSolver()
        wj_window.mainloop()
        
    def open_eight_puzzle():
        ep_window = tk.Toplevel(root)
        EightPuzzleGUI(ep_window)
    
    def open_tsp_solver():
        tsp_window = tk.Toplevel(root)
        TSPGui(tsp_window)

    ttk_button = tk.Button(root, text="Play Tic Tac Toe", command=open_tic_tac_toe)
    ttk_button.pack(pady=10)

    tsp_button = tk.Button(root, text="Play TSP Solver", command=open_tsp_solver)
    tsp_button.pack(pady=10)
    
    wj_button = tk.Button(root, text="Play Water Jug Puzzle", command=open_water_jug_solver)
    wj_button.pack(pady=10)
    
    rim_button = tk.Button(root, text="Play Rat in a Maze", command=open_rat_in_a_maze)
    rim_button.pack(pady=10)
    
    ep_button = tk.Button(root, text="Play 8 Puzzle Game", command=open_eight_puzzle)
    ep_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
