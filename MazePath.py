import numpy as np
from PIL import Image
import random
import csv

def maze_from_csv(path): # importuje jego bludiště jako bool
    raw_matrix = np.loadtxt(path, delimiter=",", dtype=int)
    maze = raw_matrix.astype(bool)
    return maze

def adj_matrix(maze): # tvoří incidenční matici (kiero obsahuje info, kiere dwa okiynka, kiere ze sebóm sąsiadujóm sóm průchozí)
    height, width = maze.shape
    total_cells = height * width

    adjacency = np.zeros((total_cells, total_cells), dtype=int)

    for y in range(height):
        for x in range(width):
            if maze[y][x] == False:
                current = y * width + x
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if maze[ny][nx] == False:
                            neighbor = ny * width + nx
                            adjacency[current][neighbor] = 1
    return adjacency

def shortest_path(adjacency, start, end):
    queue = [(start, [start])]
    visited = set()

    while queue:
        current, path = queue.pop(0)
        if current == end:
            return path
        visited.add(current)
        for neighbor in range(len(adjacency)):
            if adjacency[current][neighbor] == 1 and neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None

def maze_with_path(maze, path):
    height, width = maze.shape
    img = Image.new("RGB", (width, height))

    for y in range(height):
        for x in range(width):
            color = (0,0,0) if maze[y][x] else (255, 255, 255)
            img.putpixel((x,y), color)

    for index in path:
        y = index // width
        x = index % width
        img.putpixel((x,y), (255,0,0))

    img = img.resize((width*20, height*20), Image.NEAREST)
    img.save("solved.png")

def generate_maze(n, base="empty", density= 0.7):
    maze = create_template(n, base)

    cells = list((y,x) for y in range(n) for x in range(n)
                 if (y,x) not in [(0,0), (n-1, n-1)])
    random.shuffle(cells)

    for y,x in cells:
        if random.random() > density:
            continue

        maze[y][x] = True
        adj = adj_matrix(maze)
        path = shortest_path(adj, 0, n*n - 1)

        if path is None:
            maze[y][x] = False
        
    return maze

def create_template(n, base="empty"):
    maze = np.zeros((n,n), dtype=bool)

    if base == "hslalom":
        for y in range(1, n-1, 2):
            for x in range(n):
                if x != y:
                    maze[y][x] = True

    if base == "vslalom":
        for x in range(1, n-1, 2):
            for y in range(n):
                if (y-2) % n != x:
                    maze[y][x] = True
        
    if base == "sslalom":
        for y in range(1, n-1, 2):
            for x in range(n):
                if (y+10) % n != x:
                    maze[y][x] = True

    if base == "snake":
        for y in range(1, n-1, 2):
            for x in range(n):
                if y % n != (x*10) % n:
                    maze[y][x] = True

    return maze 

def solve_and_save(maze):
    n = maze.shape[0]
    adj = adj_matrix(maze)
    path = shortest_path(adj, 0, n * n - 1)

    if path is None:
        print("Bludiště je neprůchozí.")
    else:
        print(f"Cesta nalezena! Délka: {len(path)} kroků")
        maze_with_path(maze, path)

maze = maze_from_csv("data/maze_3.csv")
maze1 = create_template(41, base="empty")
solve_and_save(maze1)