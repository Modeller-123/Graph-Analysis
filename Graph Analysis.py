###############################
###############################
###                         ###
###         IMPORTS         ###
###                         ###
###############################
###############################

from tkinter import *
import numpy as np
from itertools import permutations
import sys

# Set high recursion limit for analysis
sys.setrecursionlimit(100000)


###############################
###############################
###                         ###
###     GRAPH ANALYSIS      ###
###                         ###
###############################
###############################

class Graph(object):
    """
    A simple undirected graph implementation with various graph-theoretic 
    algorithms, including adjacency/Hashimoto matrix creation, degree 
    calculations, centrality measures, path finding, chromatic coloring, 
    and component detection.

    Vertices are labeled 1 through n.
    Edges are stored as unordered pairs (tuples) of vertices.
    """

    def __init__(self, n, edges):
        """
        Initialize a graph with n vertices and a set of edges.

        Args:
            n (int): Number of vertices.
            edges (list[tuple[int, int]]): List of edges represented as vertex pairs.
        """
        self.vertices = []          # List of vertices [1, 2, ..., n]
        self.V = 0                  # Number of vertices
        self.add_vertices(n)

        self.edges = []             # List of edges
        self.E = 0                  # Number of edges
        self.add_edges(edges)

        self.adj_matrix = np.array([])  # Adjacency matrix
        self.deg_seq = []               # Degree sequence
        self.hash_matrix = []           # Hashimoto (non-backtracking) matrix

    def add_vertices(self, n):
        """Add n vertices to the graph."""
        for i in range(1, n+1):
            self.vertices.append(self.V + i)
        self.V += n

    def add_edges(self, edge_set):
        """Add edges from a given edge set (avoiding duplicates)."""
        for edge in edge_set:
            if edge not in self.edges and (edge[1], edge[0]) not in self.edges:
                self.edges.append(edge)
                self.E += 1

    def create_adjacency_matrix(self):
        """Construct the adjacency matrix of the graph."""
        self.adj_matrix = []
        for i in range(1, self.V+1):
            row = []
            for j in range(1, self.V+1):
                row.append(1 if (i, j) in self.edges or (j, i) in self.edges else 0)
            self.adj_matrix.append(row)
        self.adj_matrix = np.array(self.adj_matrix)

    def create_degree_sequence(self):
        """Generate the degree sequence (list of vertex degrees)."""
        self.deg_seq = [0] * self.V
        for edge in self.edges:
            self.deg_seq[edge[0]-1] += 1
            self.deg_seq[edge[1]-1] += 1

    def degree_information(self):
        """
        Returns:
            tuple (min_degree, max_degree, avg_degree)
        """
        return (min(self.deg_seq), max(self.deg_seq), sum(self.deg_seq)/self.V)

    def find_neighbours(self, i):
        """Return the list of neighbours of vertex i."""
        neighbours = []
        for e in self.edges:
            if i == e[0] and e[1] not in neighbours:
                neighbours.append(e[1])
            elif i == e[1] and e[0] not in neighbours:
                neighbours.append(e[0])
        return neighbours

    def num_walks(self, l, i, j, *args):
        """
        Find the number of walks of length l from vertex i to j using adjacency matrix powers.
        If an adjacency matrix is passed in args, use that; otherwise use self.adj_matrix.
        """
        A = args[0] if len(args) == 1 else self.adj_matrix
        if l == 0:
            return (np.identity(self.V)[i-1][j-1])
        A_power_l = A
        for _ in range(l-1):
            A_power_l = np.matmul(A_power_l, A)
        return A_power_l[i-1][j-1]

    def find_katz_centrality(self, alpha):
        """Compute Katz centrality with parameter alpha."""
        return np.matmul(np.linalg.inv(np.identity(self.V) - alpha * self.adj_matrix), np.array([1] * self.V))

    def find_page_rank(self, alpha):
        """Compute PageRank vector with damping factor alpha (Google uses 0.85)."""
        diagonal = np.array([[0] * self.V] * self.V)
        self.create_degree_sequence()
        for i in range(self.V):
            diagonal[i][i] = self.deg_seq[i]
        return np.matmul(np.matmul(diagonal, np.linalg.inv(diagonal - alpha * self.adj_matrix)), np.array([1] * self.V))

    def create_hashimoto_matrix(self):
        """Construct the Hashimoto (non-backtracking) matrix."""
        self.hash_matrix = []
        for i in range(2*len(self.edges)):
            e1 = self.edges[i-len(self.edges)][::-1] if i >= len(self.edges) else self.edges[i]
            row = []
            for j in range(2*len(self.edges)):
                e2 = self.edges[j-len(self.edges)][::-1] if j >= len(self.edges) else self.edges[j]
                row.append(1 if (e1[1] == e2[0]) and (e1[0] != e2[1]) else 0)
            self.hash_matrix.append(row)
        self.hash_matrix = np.array(self.hash_matrix)

    def num_nb_walks(self, l, e1, e2):
        """Find the number of non-backtracking walks of length l+1 from edge e1 to e2."""
        if e1 in self.edges:
            i = self.edges.index(e1)
        else:
            i = self.edges.index((e1[1], e1[0])) + len(self.edges)
        if e2 in self.edges:
            j = self.edges.index(e2)
        else:
            j = self.edges.index((e2[1], e2[0])) + len(self.edges)

        H_power_l = self.hash_matrix
        for _ in range(l-1):
            H_power_l = np.matmul(H_power_l, self.hash_matrix)
        return H_power_l[i][j]

    def distance(self, i, j):
        """Compute the shortest path length (distance) between vertices i and j."""
        for l in range(self.V+1):
            if self.num_walks(l, i, j) > 0:
                return l
        return None

    def diameter(self):
        """Compute the diameter of the graph (longest shortest path)."""
        maximum = 0
        coords = []
        for i in self.vertices:
            for j in self.vertices:
                l = self.distance(i, j)
                if l is None:
                    return ["Infinity", []]
                if l > maximum:
                    maximum, coords = l, [i, j]
        return [maximum, coords]

    def find_path(self, i, j, l, possibilities):
        """Find a path of exact length l between vertices i and j using backtracking."""
        neighbours = self.find_neighbours(i)
        if l == 1 and j in neighbours:
            return [item[0] for item in possibilities] + [j]

        use = [item[0] for item in possibilities]
        candidates = [v for v in neighbours if v != j and v not in use]

        if not candidates:  # backtrack
            if len(possibilities) == 1:
                return None
            possibilities[-1] = possibilities[-1][1:]
            while not possibilities[-1]:
                possibilities.pop()
                if len(possibilities) == 1:
                    return None
                possibilities[-1] = possibilities[-1][1:]
                l += 1
            return self.find_path(possibilities[-1][0], j, l, possibilities)
        else:  # step forward
            possibilities.append(candidates)
            return self.find_path(possibilities[-1][0], j, l-1, possibilities)

    def hamiltonian_path(self):
        """Attempt to find a Hamiltonian path in the graph (if one exists)."""
        for i in range(self.V):
            v1 = self.vertices[i]
            for j in range(i):
                v2 = self.vertices[j]
                path = self.find_path(v1, v2, self.V-1, [[v1]])
                if path:
                    return path
        return None

    def shortest_path(self, i, j):
        """Return the shortest path (list of vertices) between vertices i and j."""
        d = self.distance(i, j)
        if d is None:
            return None
        elif i == j:  # cycle
            for l in range(3, self.V+1):
                for n in self.find_neighbours(i):
                    path = self.find_path(i, n, l-1, [[i]])
                    if path:
                        return path + [i]
            return None
        else:
            return self.find_path(i, j, d, [[i]])

    def girth(self):
        """Find the girth (length of shortest cycle) of the graph."""
        path = []
        for v in self.vertices:
            try_path = self.shortest_path(v, v)
            if try_path and (not path or len(try_path) < len(path)):
                path = try_path
                if len(path)-1 == 3:  # triangle found
                    break
        return [len(path)-1, path]

    def circumference(self):
        """Find the circumference (length of the longest cycle) of the graph."""
        path = []
        for l in range(self.V, 2, -1):
            for v in self.vertices:
                for n in self.find_neighbours(v):
                    path = self.find_path(v, n, l-1, [[v]])
                    if path:
                        return [len(path), path + [v]]
        return [0, []]

    def greedy_algorithm(self, *args):
        """Perform greedy coloring of the graph. Returns color assignment list."""
        colours = [0] * self.V
        colours[0] = 1
        order = [i for i in range(1, self.V+1)] if not args else args[0]
        for i in order:
            used_colours = {colours[n-1] for n in self.find_neighbours(i)}
            c = 1
            while c in used_colours:
                c += 1
            colours[i-1] = c
        return colours

    def chromatic_colouring(self):
        """Find chromatic coloring using all possible vertex orderings (brute force for small graphs)."""
        use = min(self.V, 8)
        orderings = list(permutations(range(1, use+1)))
        if self.V > 8:
            leftover = tuple(range(9, self.V+1))
            orderings = [o + leftover for o in orderings]

        k = self.V
        best = [i for i in range(1, self.V+1)]
        for order in orderings:
            colours = self.greedy_algorithm(order)
            l = len(set(colours))
            if l < k:
                k, best = l, colours
        return best

    def find_connected_components(self):
        """Find connected components of the graph."""
        components = []
        for v1 in range(1, self.V+1):
            if any(v1 in comp for comp in components):
                continue
            connected = []
            for v2 in range(1, self.V+1):
                for l in range(1, self.V):
                    if self.num_walks(l, v1, v2):
                        connected.append(v2)
                        break
            components.append(connected)
        return components


###############################
###############################
###                         ###
###     GRAPH INPUT GUI     ###
###                         ###
###############################
###############################
    

class GraphApp:
    """
    A Tkinter-based interactive graph editor application.

    Features:
        - Add vertices (right-click).
        - Add edges between vertices (left-click and drag).
        - Remove vertices or edges (double-click).
        - Clear the canvas.
        - Generate special graphs (star, complete).
        - Optionally color vertices.

    Attributes:
        root (Tk): Tkinter root window.
        canvas (Canvas): Main drawing canvas.
        frame (Frame): Side control panel frame.
        vertices (list): List of vertex positions as (x, y) tuples.
        edges (list): List of edges represented as (v1, v2) index pairs.
        mode (str): Interaction mode ("edit" or "colour").
        chosen_colour (int): Index for the currently chosen vertex color.
        labels (list): List of label widgets for instructions and settings.
        buttons (list): List of button widgets for actions.
        edit_enter_but (Button): Button to "enter" the graph (disabled until a vertex exists).
    """

    def __init__(self, root):
        """
        Initialize the GraphApp GUI, bind events, and configure controls.

        Args:
            root (Tk): The Tkinter root window.
        """
        # Core state variables
        self.root = root
        self.vertices = []          # Stores vertex coordinates
        self.edges = []             # Stores edge connections
        self.mode = "edit"          # Default mode is editing
        self.chosen_colour = 1      # Default color index

        # Canvas for drawing the graph
        self.canvas = Canvas(self.root)
        self.canvas.config(background="white", highlightbackground="grey80", highlightthickness=2)
        self.canvas.pack(fill="both", expand=True)

        # Mouse bindings for interaction
        self.canvas.bind("<Button-3>", self.addVertex)          # Right click: add vertex
        self.canvas.bind("<Button-1>", self.addEdge)            # Left click: start edge
        self.canvas.bind("<ButtonRelease-1>", self.addEdge)     # Release: complete edge
        self.canvas.bind("<B1-Motion>", self.movingLine)        # Drag: show temporary edge
        self.canvas.bind("<Double-Button-1>", self.removal)     # Double click: remove

        # Side panel frame for controls
        self.frame = Frame(self.root, width=300, height=450, highlightbackground="grey80", highlightthickness=2, highlightcolor="grey80")
        self.frame.pack(anchor=NE)
        self.frame.grid_propagate(False)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=3)
        self.frame.rowconfigure(2, weight=3)
        self.frame.rowconfigure(3, weight=3)
        self.frame.rowconfigure(4, weight=1)

        # Place frame inside canvas so it resizes properly
        self.window = self.canvas.create_window(500, 0, window=self.frame, anchor="nw", tag="window")
        self.canvas.bind("<Configure>", self.handle_configure)

        # Labels and instructions
        l1 = Label(self.frame, width=40)
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, text="Graph Editing", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)
        l3 = Label(self.frame, text="""
        - Right click to add a vertex

        - Left click and drag from one vertex
          to another to add an edge

        - Double click a vertex or edge to
          remove it
        """, justify="left", font=('Arial 12'))
        l3.grid(row=1, column=0, sticky=N, columnspan=2)

        # Buttons for graph control
        b1 = Button(self.frame, text="Clear Canvas", font=('Arial 12'), bg="white", command=self.clear_canvas)
        b1.grid(row=1, column=0, sticky=S, columnspan=2)
        l4 = Label(self.frame, text="Size (1-16)", font=('Arial 12'))
        l4.grid(row=2, column=0, sticky=S)
        size_entry = Entry(self.frame, width=4, font=('Arial 12'))
        size_entry.grid(row=3, column=0, sticky=N)
        b2 = Button(self.frame, text="Star Graph", font=('Arial 12'), bg="white", command=lambda: self.make_star_graph(size_entry))
        b2.grid(row=2, column=1, sticky=S)
        b3 = Button(self.frame, text="Complete Graph", font=('Arial 12'), bg="white", command=lambda: self.make_complete_graph(size_entry))
        b3.grid(row=3, column=1, sticky=N)

        # Store UI elements for later reference
        self.labels = [l1, l2, l3, l4, size_entry]
        self.buttons = [b1, b2, b3]

        # Button to "enter" the graph (enabled only if at least one vertex exists)
        self.edit_enter_but = Button(self.frame, text="Enter Graph", font=('Arial 12'), bg="light blue",
                                     command=self.enter_graph, state=DISABLED)
        self.edit_enter_but.grid(row=4, column=0, columnspan=2)


    ################################
    ###                          ###
    ###   GRAPH EDITING OPTIONS  ###
    ###                          ###
    ################################
    def clear_canvas(self):
        """
        Clear all vertices and edges from the canvas and reset state.
        """
        # Remove vertices
        for i in range(1, len(self.vertices) + 1):
            self.canvas.delete("circle" + str(i))
            self.canvas.delete("text" + str(i))

        # Remove edges
        for e in self.edges:
            self.canvas.delete("edge:" + str(e[0]) + "," + str(e[1]))

        # Reset state
        self.vertices = []
        self.edges = []
        self.edit_enter_but.config(state=DISABLED)

    def addVertex(self, event):
        """
        Add a new vertex at the mouse position if space allows.

        Args:
            event (Event): Tkinter mouse event containing click position.
        """
        if self.mode == 'edit':
            x, y = event.x, event.y

            # Prevent overlapping with existing vertices
            overlapping = any(((x - v[0]) ** 2 + (y - v[1]) ** 2) ** 0.5 <= 35 for v in self.vertices)

            if not overlapping:
                # Draw vertex as circle and label
                self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15,
                                        fill='light blue', outline='black', width=2,
                                        tags="circle" + str(len(self.vertices) + 1))
                self.canvas.create_text(x, y, text=len(self.vertices) + 1,
                                        fill="black", font=('Arial 10 bold'),
                                        tags="text" + str(len(self.vertices) + 1))
                self.vertices.append((x, y))
                self.canvas.tag_lower("circle" + str(len(self.vertices)))  # Keep vertices under edges
                self.edit_enter_but.config(state=NORMAL)

    def addEdge(self, event):
        """
        Add an edge between two vertices, or color a vertex if in 'colour' mode.

        Args:
            event (Event): Tkinter mouse event containing click/drag position.
        """
        if self.mode == 'edit':
            # Remove temporary line if exists
            self.canvas.delete("moving_line")

            if event.state == 0:
                # Store starting point
                self.start = (event.x, event.y)
            else:
                # Finish edge creation
                finish = (event.x, event.y)
                edge = [0, 0]  # Store indices of vertices
                count = 0

                # Check if start and end clicks are on vertices
                for v in self.vertices:
                    count += 1
                    if ((self.start[0] - v[0]) ** 2 + (self.start[1] - v[1]) ** 2) ** 0.5 <= 15:
                        v1, edge[0] = v, count
                    if ((finish[0] - v[0]) ** 2 + (finish[1] - v[1]) ** 2) ** 0.5 <= 15:
                        v2, edge[1] = v, count

                edge = (edge[0], edge[1])

                # Only create valid edges (no loops, no duplicates)
                if (edge[0] != 0 and edge[1] != 0 and edge[0] != edge[1] and
                    edge not in self.edges and (edge[1], edge[0]) not in self.edges):

                    # Compute offsets so edges start/finish at circle boundary
                    diff0, diff1 = v1[0] - v2[0], v1[1] - v2[1]
                    if diff1 == 0: diff1 = 0.00001
                    if diff0 == 0: diff0 = 0.00001
                    theta = np.arctan(abs(diff1) / abs(diff0))

                    use_start = (v1[0] - diff0 / abs(diff0) * 15 * np.cos(theta),
                                 v1[1] - diff1 / abs(diff1) * 15 * np.sin(theta))
                    use_finish = (v2[0] + diff0 / abs(diff0) * 15 * np.cos(theta),
                                  v2[1] + diff1 / abs(diff1) * 15 * np.sin(theta))

                    # Draw edge line
                    self.canvas.create_line(use_start[0], use_start[1], use_finish[0], use_finish[1],
                                            tags="edge:" + str(edge[0]) + "," + str(edge[1]), width=3)
                    self.edges.append(edge)

        elif self.mode == "colour":
            # Change vertex color if clicked
            colour_ops = ["red3", "blue", "green4", "goldenrod", "magenta3", "SeaGreen3", "dark olive green", "DarkOrange3"]
            col = self.chosen_colour.get()
            for count, v in enumerate(self.vertices, start=1):
                if ((event.x - v[0]) ** 2 + (event.y - v[1]) ** 2) ** 0.5 <= 15:
                    self.canvas.itemconfigure("circle" + str(count), fill=colour_ops[col - 1])

    def movingLine(self, event):
        """
        Show a temporary line while dragging to create an edge.

        Args:
            event (Event): Tkinter mouse drag event.
        """
        if self.mode == 'edit':
            v1 = None
            for v in self.vertices:
                if ((self.start[0] - v[0]) ** 2 + (self.start[1] - v[1]) ** 2) ** 0.5 <= 15:
                    v1 = v
                    break

            if v1:
                self.canvas.delete("moving_line")
                self.canvas.create_line(v1[0], v1[1], event.x, event.y, tag="moving_line", width=3)
                self.canvas.tag_lower("moving_line")

    def removal(self, event):
        """
        Remove a vertex (and its edges) or an edge, depending on click position.

        Args:
            event (Event): Tkinter mouse double-click event.
        """
        if self.mode == 'edit':
            remove = None

            # First check if a vertex is clicked
            for count, v in enumerate(self.vertices, start=1):
                if ((event.x - v[0]) ** 2 + (event.y - v[1]) ** 2) ** 0.5 <= 15:
                    # Remove vertex circle and text
                    self.canvas.delete("circle" + str(count))
                    self.canvas.delete("text" + str(count))
                    remove = count
                elif remove:
                    # Re-number subsequent vertices
                    self.canvas.itemconfigure("text" + str(count), text=str(count - 1), tag="text" + str(count - 1))
                    self.canvas.itemconfigure("circle" + str(count), tag="circle" + str(count - 1))

            if remove:
                # Remove from state
                self.vertices.remove(self.vertices[remove - 1])

                # Remove edges connected to removed vertex
                remove_edges = [edge for edge in self.edges if remove in edge]
                for edge in remove_edges:
                    self.canvas.delete("edge:" + str(edge[0]) + "," + str(edge[1]))
                    self.edges.remove(edge)

                # Re-number edge tags
                for i in range(len(self.edges)):
                    if self.edges[i][0] > remove:
                        self.canvas.itemconfigure("edge:" + str(self.edges[i][0]) + "," + str(self.edges[i][1]),
                                                  tag="edge:" + str(self.edges[i][0] - 1) + "," + str(self.edges[i][1]))
                        self.edges[i] = (self.edges[i][0] - 1, self.edges[i][1])
                    if self.edges[i][1] > remove:
                        self.canvas.itemconfigure("edge:" + str(self.edges[i][0]) + "," + str(self.edges[i][1]),
                                                  tag="edge:" + str(self.edges[i][0]) + "," + str(self.edges[i][1] - 1))
                        self.edges[i] = (self.edges[i][0], self.edges[i][1] - 1)
            else:
                # Otherwise, check if an edge was clicked
                for edge in self.edges:
                    v1, v2 = self.vertices[edge[0] - 1], self.vertices[edge[1] - 1]
                    A, B = v2[1] - v1[1], -(v2[0] - v1[0])
                    C = -v1[1] * B - v1[0] * A
                    distance = abs(A * event.x + B * event.y + C) / (A ** 2 + B ** 2) ** 0.5

                    # Check distance from line and bounding box
                    if (distance < 6 and
                        min(v1[0], v2[0]) - 5 <= event.x <= max(v1[0], v2[0]) + 5 and
                        min(v1[1], v2[1]) - 5 <= event.y <= max(v1[1], v2[1]) + 5):
                        self.canvas.delete("edge:" + str(edge[0]) + "," + str(edge[1]))
                        self.edges.remove(edge)
                        break

            # Disable "enter" button if no vertices remain
            if not self.vertices:
                self.edit_enter_but.config(state=DISABLED)

    def make_star_graph(self, size_entry):
        """
        Generate and display a star graph with a central vertex connected to all others.

        Args:
            size_entry (Entry): Tkinter entry widget containing the number of outer vertices (1–16).
                                The actual number of vertices created is size_entry + 1
                                (including the center).
        """
        correct_value = 1
        try:
            n = int(size_entry.get()) + 1  # +1 for central vertex
            if not (1 <= n <= 17):
                correct_value = 0
        except:
            correct_value = 0

        if correct_value == 1:
            # Reset the canvas and enable "Enter Graph" button
            self.clear_canvas()
            self.edit_enter_but.config(state=NORMAL)

            # Compute center of canvas
            (centre_x, centre_y) = (int(self.canvas.winfo_width() / 2) - 150,
                                    int(self.canvas.winfo_height() / 2))

            # Draw central vertex
            self.canvas.create_oval(centre_x - 15, centre_y - 15,
                                    centre_x + 15, centre_y + 15,
                                    fill='light blue', outline='black', width=2,
                                    tags="circle" + str(len(self.vertices) + 1))
            self.canvas.create_text(centre_x, centre_y, text=len(self.vertices) + 1,
                                    fill="black", font=('Arial 10 bold'),
                                    tags="text" + str(len(self.vertices) + 1))
            self.vertices.append((centre_x, centre_y))

            # Radius for outer vertices
            radius = 30 * (n - 1) / np.pi + 40
            theta = 0

            for i in range(2, n + 1):
                # Add outer vertex evenly spaced around the circle
                (x, y) = (centre_x + radius * np.cos(theta),
                          centre_y + radius * np.sin(theta))
                self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15,
                                        fill='light blue', outline='black', width=2,
                                        tags="circle" + str(len(self.vertices) + 1))
                self.canvas.create_text(x, y, text=len(self.vertices) + 1,
                                        fill="black", font=('Arial 10 bold'),
                                        tags="text" + str(len(self.vertices) + 1))
                self.vertices.append((x, y))
                theta += 2 * np.pi / (n - 1)

                # Draw edge from center to this vertex
                self.edges.append((1, i))
                self.canvas.create_line(centre_x, centre_y, x, y,
                                        tags="edge:" + str(1) + "," + str(i), width=3)
                self.canvas.tag_lower("edge:" + str(1) + "," + str(i))

    def make_complete_graph(self, size_entry):
        """
        Generate and display a complete graph (Kₙ) with all vertices interconnected.

        Args:
            size_entry (Entry): Tkinter entry widget containing the number of vertices (1–16).
        """
        correct_value = 1
        try:
            n = int(size_entry.get())
            if not (1 <= n <= 16):
                correct_value = 0
        except:
            correct_value = 0

        if correct_value == 1:
            # Reset the canvas and enable "Enter Graph" button
            self.clear_canvas()
            self.edit_enter_but.config(state=NORMAL)

            # Compute layout parameters
            radius = 30 * n / np.pi + 40
            theta = 0
            (centre_x, centre_y) = (int(self.canvas.winfo_width() / 2) - 150,
                                    int(self.canvas.winfo_height() / 2))

            for i in range(1, n + 1):
                # Add vertex evenly spaced on the circle
                (x, y) = (centre_x + radius * np.cos(theta),
                          centre_y + radius * np.sin(theta))
                self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15,
                                        fill='light blue', outline='black', width=2,
                                        tags="circle" + str(len(self.vertices) + 1))
                self.canvas.create_text(x, y, text=len(self.vertices) + 1,
                                        fill="black", font=('Arial 10 bold'),
                                        tags="text" + str(len(self.vertices) + 1))
                self.vertices.append((x, y))
                theta += 2 * np.pi / n

                # Add edges to all previously placed vertices
                for j in range(1, i):
                    self.edges.append((j, i))
                    self.canvas.create_line(self.vertices[j - 1][0], self.vertices[j - 1][1],
                                            x, y,
                                            tags="edge:" + str(j) + "," + str(i), width=3)
                    self.canvas.tag_lower("edge:" + str(j) + "," + str(i))

    def destroy_all(self, listy):
        for i in listy:
            i.destroy()

    def handle_configure(self, event):
        """
        Reposition the control frame when the canvas resizes.

        Args:
            event (Event): Tkinter configure event containing new canvas dimensions.
        """
        self.canvas.delete("window")
        self.window = self.canvas.create_window(event.width - 300, 0, window=self.frame, anchor="nw", tag="window")


    ################################
    ###                          ###
    ###        ENTER GRAPH       ###
    ###                          ###
    ################################

    def enter_graph(self):
        """
        Switch the application into 'calc' mode for analyzing the graph.
        """
        self.mode = 'calc'

        # Remove editing UI
        self.destroy_all(self.labels)
        self.destroy_all(self.buttons)

        # Change canvas appearance for analysis mode
        self.canvas.config(background="black")
        for i in range(len(self.vertices)):
            self.canvas.itemconfigure("circle" + str(i + 1), fill="blue", outline="white")
            self.canvas.itemconfigure("text" + str(i + 1), fill="white")
        for e in self.edges:
            self.canvas.itemconfigure("edge:" + str(e[0]) + "," + str(e[1]),
                                      fill="white", width=3)

        # Check whether graph drawing is planar
        self.planar = self.check_planar()

        # Update control frame appearance
        self.frame.configure(background="grey21")
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white",
                   text="Graph Properties", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)
        l3 = Label(self.frame, bg="grey21", fg="white",
                   text="Select one of the options below:\n\n"
                        "(Note that planar info is only\n"
                        "available for plane graphs)",
                   justify="left", font=('Arial 12'))
        l3.grid(row=1, column=0, sticky=N, columnspan=2)

        # Create analysis buttons
        b1 = Button(self.frame, text="Degree Info", font=('Arial 12'),
                    width=12, bg="black", fg="white", command=self.degree_info_op)
        b1.grid(row=1, column=0, sticky=S)
        b2 = Button(self.frame, text="Walks", font=('Arial 12'),
                    width=12, bg="black", fg="white", command=self.walks_op)
        b2.grid(row=1, column=1, sticky=S)
        b3 = Button(self.frame, text="Centrality", font=('Arial 12'),
                    width=12, bg="black", fg="white", command=self.centrality_op)
        b3.grid(row=2, column=0)
        b4 = Button(self.frame, text="Dimensions", font=('Arial 12'),
                    width=12, bg="black", fg="white", command=self.dimensions_op)
        b4.grid(row=2, column=1)
        b5 = Button(self.frame, text="Colouring", font=('Arial 12'),
                    width=12, bg="black", fg="white", command=self.colouring_op)
        b5.grid(row=3, column=0, sticky=N)
        b6 = Button(self.frame, text="Planar Info", font=('Arial 12'),
                    width=12, bg="black", fg="white", command=self.planar_info_op)
        b6.grid(row=3, column=1, sticky=N)

        # Track new labels and buttons
        self.labels = [l1, l2, l3]
        self.buttons = [b1, b2, b3, b4, b5, b6]

        # Disable "Planar Info" button if graph is not planar
        if not self.planar:
            b6.config(state=DISABLED)

        # Update Enter button to allow switching back to editing
        self.edit_enter_but.config(text="Edit Graph", bg="blue",
                                   fg="white", command=self.edit_graph)
        

    ################################
    ###                          ###
    ###         EDIT GRAPH       ###
    ###                          ###
    ################################
    def edit_graph(self):
        self.mode = 'edit'

        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Create user interface
        self.canvas.config(background="white")
        for i in range (0,len(self.vertices)):
            self.canvas.itemconfigure("circle"+str(i+1), fill="light blue", outline="black")
            self.canvas.itemconfigure("text"+str(i+1), fill="black")
        for e in self.edges:
            self.canvas.itemconfigure("edge:"+str(e[0])+","+str(e[1]),fill="black")
        self.frame.configure(background="grey94")
        l1 = Label(self.frame, width=40)
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, text="Graph Editing", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)
        l3 = Label(self.frame, text="""
        - Right click to add a vertex

        - Left click and drag from one vertex
          to another to add an edge

        - Double click a vertex or edge to
          remove it
        """, justify="left", font=('Arial 12'))
        l3.grid(row=1, column=0, sticky=N, columnspan=2)
        b1 = Button(self.frame, text="Clear Canvas", font=('Arial 12'), bg="white", command=self.clear_canvas)
        b1.grid(row=1, column=0, sticky=S, columnspan=2)
        l4 = Label(self.frame, text="Size (1-16)", font=('Arial 12'))
        l4.grid(row=2, column=0, sticky=S)
        size_entry = Entry(self.frame, width=4, font=('Arial 12'))
        size_entry.grid(row=3, column=0, sticky=N)
        b2 = Button(self.frame, text="Star Graph", font=('Arial 12'), bg="white", command=lambda:self.make_star_graph(size_entry))
        b2.grid(row=2, column=1, sticky=S)
        b3 = Button(self.frame, text="Complete Graph", font=('Arial 12'), bg="white", command=lambda:self.make_complete_graph(size_entry))
        b3.grid(row=3, column=1, sticky=N)

        self.labels = [l1,l2,l3,l4,size_entry]
        self.buttons = [b1,b2,b3]
        
        self.edit_enter_but.config(text="Enter Graph", bg="light blue", fg="black", command=self.enter_graph)



    ################################
    ###                          ###
    ###  GRAPH ANALYSIS OPTIONS  ###
    ###                          ###
    ################################

    def check_planar(self):
        """
        Check whether the current graph drawing is planar (no edge crossings).

        Returns:
            bool: True if graph is planar, False otherwise.
        """
        planar = True
        for i in range(len(self.edges)):
            e1 = self.edges[i]
            v1 = self.vertices[e1[0] - 1]
            v2 = self.vertices[e1[1] - 1]

            # Step 1: Check that no other vertex lies "inside" this edge
            for v in self.vertices:
                if v != v1 and v != v2:
                    diff1 = v2[0] - v1[0]
                    if diff1 == 0:
                        diff1 = 0.0001  # avoid divide by zero
                    # Distance of point v from line (v1,v2)
                    if (abs(v[1] - v1[1] - (v[0] - v1[0]) * (v2[1] - v1[1]) / diff1) < 6
                        and (min(v1[0], v2[0]) - 5 < v[0] < max(v1[0], v2[0]) + 5)
                        and (min(v1[1], v2[1]) - 5 < v[1] < max(v1[1], v2[1]) + 5)):
                        planar = False
                        break
            if not planar:
                break

            # Step 2: Check for edge intersections with all previous edges
            for j in range(i):
                e2 = self.edges[j]
                w1 = self.vertices[e2[0] - 1]
                w2 = self.vertices[e2[1] - 1]

                # Only check if edges don't share endpoints
                if v1 != w1 and v1 != w2 and v2 != w1 and v2 != w2:
                    diff1 = v2[0] - v1[0]
                    diff2 = w2[0] - w1[0]
                    if diff1 == 0:
                        diff1 = 0.0001
                    if diff2 == 0:
                        diff2 = 0.0001
                    m1 = (v2[1] - v1[1]) / diff1
                    m2 = (w2[1] - w1[1]) / diff2
                    diff3 = m1 - m2
                    if diff3 == 0:
                        diff3 = 0.0001
                    c1 = v1[1] - m1 * v1[0]
                    c2 = w1[1] - m2 * w1[0]
                    int_x = (c2 - c1) / diff3
                    int_y = m1 * int_x + c1

                    # Check if intersection lies within both line segments
                    if (min(v1[0], v2[0]) - 5 < int_x < max(v1[0], v2[0]) + 5
                        and min(v1[1], v2[1]) - 5 < int_y < max(v1[1], v2[1]) + 5
                        and min(w1[0], w2[0]) - 5 < int_x < max(w1[0], w2[0]) + 5
                        and min(w1[1], w2[1]) - 5 < int_y < max(w1[1], w2[1]) + 5):
                        planar = False
                        break
            if not planar:
                break
        return planar

    def degree_info_op(self):
        """
        Sets up the UI to display degree information of the graph.
        Shows minimum, maximum, and average degrees, as well as
        adjacency matrix and degree sequence for small graphs (≤16 vertices).
        """
        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Title bar and heading
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white", text="Degree Information", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)

        # Initialize graph and compute adjacency matrix and degree sequence
        graph = Graph(len(self.vertices), self.edges)
        graph.create_adjacency_matrix()
        graph.create_degree_sequence()

        # Sort degree sequence descending
        degree_sequence = graph.deg_seq.copy()
        degree_sequence.sort(reverse=True)

        # Compute min, max, and average degrees
        minimum, maximum, average = graph.degree_information()

        # Display degree statistics
        l3 = Label(self.frame, bg="grey21", fg="white",
                   text=f"Minimum degree: {minimum}\nMaximum degree: {maximum}\nAverage degree: {round(average,2)}",
                   font=('Arial 12'))
        l3.grid(row=2, column=0, columnspan=2)

        # Display adjacency matrix if graph is small enough
        if len(self.vertices) <= 16:
            l4 = Label(self.frame, bg="grey21", fg="white", text=graph.adj_matrix, font=('Arial 10'))
            l4.grid(row=1, column=0, columnspan=2)
        else:
            l4 = Label(self.frame, bg="grey21", fg="white",
                       text="Adjacency Matrix and degree\nsequence can only be shown for\nnumber of vertices at most 16.",
                       font=('Arial 12'))
            l4.grid(row=1, column=0, columnspan=2)

        # Display degree sequence, split if necessary
        if 10 < len(degree_sequence) <= 16:
            l5 = Label(self.frame, bg="grey21", fg="white",
                       text=str(degree_sequence[:10]) + "\n" + str(degree_sequence[10:]), font=('Arial 10'))
        elif len(degree_sequence) <= 10:
            l5 = Label(self.frame, bg="grey21", fg="white", text=str(degree_sequence), font=('Arial 10'))
        else:
            l5 = Label(self.frame, bg="grey21")  # Empty placeholder for large graphs
        l5.grid(row=3, column=0, columnspan=2)

        # Store references for cleanup
        self.labels = [l1, l2, l3, l4, l5]
        self.buttons = []

        # Back button
        self.edit_enter_but.config(text="Back", command=self.enter_graph)


    def highlight_path(self, path):
        """
        Highlights a given path on the canvas using red for vertices and edges.
        
        Parameters:
            path (list): Ordered list of vertices forming the path
        """
        for i in range(len(path)-1):
            v1, v2 = path[i], path[i+1]
            self.canvas.itemconfigure("circle"+str(v1), fill="red3", outline="red4")
            self.canvas.itemconfigure("edge:"+str(v1)+","+str(v2), fill="red3", width=7)
            self.canvas.itemconfigure("edge:"+str(v2)+","+str(v1), fill="red3", width=7)
        self.canvas.itemconfigure("circle"+str(path[-1]), fill="red3", outline="red4")


    def reset_colours(self):
        """
        Resets all vertex and edge colours to default.
        Vertices become blue with white text, edges become white.
        """
        for i in range(len(self.vertices)):
            self.canvas.itemconfigure("circle"+str(i+1), fill="blue", outline="white")
            self.canvas.itemconfigure("text"+str(i+1), fill="white")
        for e in self.edges:
            self.canvas.itemconfigure("edge:"+str(e[0])+","+str(e[1]), fill="white", width=3)


    def display_hamiltonian(self, graph, b3):
        """
        Highlights a Hamiltonian path if it exists; otherwise displays a message.
        
        Parameters:
            graph (Graph): Graph object
            b3 (Button): Reference to Hamiltonian Path button
        """
        self.reset_colours()
        path = graph.hamiltonian_path()
        if len(graph.vertices) == 1:
            path = [1]
        if path is None:
            self.buttons.remove(b3)
            l6 = Label(self.frame, text="No Hamiltonian Path", bg="grey21", fg="white", font=('Arial 11'))
            l6.grid(row=2, column=0, columnspan=2)
            self.labels.append(l6)
            b3.destroy()
        else:
            self.highlight_path(path)


    def display_num_walks(self, graph, e1, e2, e3, l7):
        """
        Calculates and displays the number of walks of given length between two vertices.
        
        Parameters:
            graph (Graph): Graph object
            e1, e2, e3 (Entry): Tkinter entries for start vertex, end vertex, and length
            l7 (Label): Label to display the result
        """
        l = e3.get()
        i = e1.get()
        j = e2.get()
        possible = True

        try:
            l = int(l)
            if l < 0: possible = False
            i = int(i)
            j = int(j)
            if not (1 <= i <= graph.V) or not (1 <= j <= graph.V):
                possible = False
        except ValueError:
            possible = False

        if possible:
            num = graph.num_walks(l, i, j)
            l7.config(text=int(num))
        else:
            l7.config(text="Invalid Inputs")


    def display_walk(self, graph, e1, e2, e3, l8):
        """
        Finds and highlights a specific walk of given length between two vertices.
        If no valid walk exists, displays an error message.
        
        Parameters:
            graph (Graph): Graph object
            e1, e2, e3 (Entry): Tkinter entries for start vertex, end vertex, and length
            l8 (Label): Label to display message if no walk exists
        """
        l = e3.get()
        i = e1.get()
        j = e2.get()
        possible = True
        self.reset_colours()

        try:
            l = int(l)
            if not (0 <= l <= graph.E):
                possible = False
            i = int(i)
            j = int(j)
            if not (1 <= i <= graph.V) or not (1 <= j <= graph.V):
                possible = False
        except ValueError:
            possible = False

        path = None
        if possible:
            l8.config(text="")
            if i == j and l == 0:
                path = [i]
            elif i == j:
                if l == 2:
                    path = None
                else:
                    for n in graph.find_neighbours(i):
                        path = graph.find_path(i, n, l-1, [[i]])
                        if path is not None:
                            path += [i]
                            break
            else:
                path = graph.find_path(i, j, l, [[i]])

            if path is not None:
                self.highlight_path(path)
            else:
                l8.config(text="No Path")
        else:
            l8.config(text="Invalid Inputs")

    def walks_op(self):
        """
        Sets up the UI for exploring walks in the graph.
        Allows users to compute the number of walks of a given length between vertices,
        display the actual walk, and check for Hamiltonian paths.
        """
        # Clear previous UI elements
        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Title bar and heading
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white", text="Walks", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)

        # Initialize graph and adjacency/Hashimoto matrices
        graph = Graph(len(self.vertices), self.edges)
        graph.create_adjacency_matrix()
        graph.create_hashimoto_matrix()

        # Inputs: start vertex, end vertex, and walk length
        l3 = Label(self.frame, text="Starting Vertex", bg="grey21", fg="white", font=('Arial 11'))
        l3.grid(row=1, column=0, sticky=N)
        e1 = Entry(self.frame, width=3, font=('Arial 11'))
        e1.grid(row=1, column=1, sticky=NW)

        l4 = Label(self.frame, text="Finishing Vertex", bg="grey21", fg="white", font=('Arial 11'))
        l4.grid(row=1, column=0)
        e2 = Entry(self.frame, width=3, font=('Arial 11'))
        e2.grid(row=1, column=1, sticky=W)

        l5 = Label(self.frame, text="Length", bg="grey21", fg="white", font=('Arial 11'))
        l5.grid(row=1, column=0, sticky=S)
        e3 = Entry(self.frame, width=3, font=('Arial 11'))
        e3.grid(row=1, column=1, sticky=SW)

        # Button for Hamiltonian path
        b3 = Button(self.frame, text="Hamiltonian Path", bg="black", fg="white",
                    font=('Arial 15 bold'), command=lambda:self.display_hamiltonian(graph, b3))
        b3.grid(row=2, column=0, columnspan=2)

        # Labels for displaying results
        l7 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l7.grid(row=3, column=1, sticky=N)
        l8 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l8.grid(row=3, column=1)
        l9 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'), width=10)
        l9.grid(row=3, column=1)

        # Buttons for number of walks and show path
        b1 = Button(self.frame, text="Number of Walks", bg="black", fg="white", font=('Arial 11'),
                    command=lambda:self.display_num_walks(graph, e1, e2, e3, l7))
        b1.grid(row=3, column=0, sticky=N)

        b2 = Button(self.frame, text="Show Path", bg="black", fg="white", font=('Arial 11'),
                    command=lambda:self.display_walk(graph, e1, e2, e3, l8))
        b2.grid(row=3, column=0)

        # Store references for cleanup
        self.labels = [l1, l2, l3, l4, l5, l7, l8, l9]
        self.buttons = [b1, b2, b3, e1, e2, e3]

        # Add back button
        self.edit_enter_but.config(text="Back", command=self.enter_graph)


    def show_centrality(self, centrality_type, graph, label, e1):
        """
        Computes and displays the selected centrality measure for the graph.
        Supports Katz centrality ('K') and PageRank ('P').
        
        Parameters:
            centrality_type (str): Either 'K' (Katz) or 'P' (PageRank)
            graph (Graph): The graph object
            label (Label): Tkinter label to display results
            e1 (Entry): Tkinter entry widget containing the alpha parameter
        """
        alpha = e1.get()
        possible = True
        try:
            alpha = float(alpha)
        except ValueError:
            possible = False

        if possible:
            if graph.V <= 16:  # Limit graph size for computation
                vector = None
                try:
                    if centrality_type == "K":
                        vector = graph.find_katz_centrality(alpha)
                    elif centrality_type == "P":
                        vector = graph.find_page_rank(alpha)
                except:
                    label.config(text="Singular Matrix")
                    return

                # Display centrality vector rounded to 3 decimal places
                text = "\n".join([str(round(v, 3)) for v in vector])
                label.config(text=text)
            else:
                label.config(text="Graph must\nbe of max\nsize 16")


    def centrality_op(self):
        """
        Sets up the UI for centrality calculations.
        Allows users to input alpha and compute Katz centrality or PageRank.
        """
        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Create graph
        graph = Graph(len(self.vertices), self.edges)
        graph.create_adjacency_matrix()
        graph.create_degree_sequence()

        # UI title and heading
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white", text="Centrality", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)

        # Alpha input
        l3 = Label(self.frame, text="Alpha", bg="grey21", fg="white", font=('Arial 11'))
        l3.grid(row=2, column=0)
        e1 = Entry(self.frame, width=5, font=('Arial 11'))
        e1.grid(row=2, column=1)

        # Placeholder labels for displaying results
        l4 = Label(self.frame, bg="grey21", height=18)
        l4.grid(row=1, column=0, columnspan=2)
        l5 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l5.grid(row=1, column=0)
        l6 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l6.grid(row=1, column=1)

        # Buttons for centrality types
        b1 = Button(self.frame, text="Katz Centrality", bg="black", fg="white", font=('Arial 11'),
                    command=lambda:self.show_centrality("K", graph, l5, e1))
        b1.grid(row=3, column=0, sticky=N)

        b2 = Button(self.frame, text="Page Rank", bg="black", fg="white", font=('Arial 11'),
                    command=lambda:self.show_centrality("P", graph, l6, e1))
        b2.grid(row=3, column=1, sticky=N)

        # Store references for cleanup
        self.labels = [l1, l2, l3, l4, l5, l6]
        self.buttons = [b1, b2, e1]

        # Add back button
        self.edit_enter_but.config(text="Back", command=self.enter_graph)


    def show_measurement(self, measurement, graph, label):
        """
        Displays the selected measurement of the graph on the label.
        
        Parameters:
            measurement (str): 'D' (Diameter), 'G' (Girth), or 'C' (Circumference)
            graph (Graph): Graph object
            label (Label): Tkinter label to display the measurement
        """
        self.reset_colours()

        if measurement == "D":  # Diameter
            if graph.V == 1:
                l = 0
                self.highlight_path(graph.vertices)
            else:
                [l, x] = graph.diameter()
                if l != "Infinity":
                    [i, j] = x
                    path = graph.find_path(i, j, l, [[i]])
                    if path is not None:
                        self.highlight_path(path)
            label.config(text=l)

        elif measurement == "G":  # Girth
            [l, path] = graph.girth()
            text = "No Cycles" if path == [] else l
            if path != []:
                self.highlight_path(path)
            label.config(text=text)

        elif measurement == "C":  # Circumference
            [l, path] = graph.circumference()
            text = "No Cycles" if path == [] else l
            if path != []:
                self.highlight_path(path)
            label.config(text=text)


    def show_distance(self, graph, l8, e1, e2):
        """
        Displays the shortest path distance between two vertices.
        Highlights the path on the canvas if it exists.

        Parameters:
            graph (Graph): The graph object
            l8 (Label): Tkinter label to show the distance
            e1, e2 (Entry): Tkinter entries containing start and end vertices
        """
        i = e1.get()
        j = e2.get()
        possible = True
        self.reset_colours()

        try:
            i = int(i)
            j = int(j)
            if not(1 <= i <= graph.V) or not(1 <= j <= graph.V):
                possible = False
        except ValueError:
            possible = False

        if possible:
            l8.config(text="")
            path = [i] if i == j else graph.shortest_path(i, j)
            if path is not None:
                self.highlight_path(path)
                l8.config(text=len(path)-1)
            else:
                l8.config(text="Infinity")
        else:
            l8.config(text="Invalid Inputs")

    def dimensions_op(self):
        """
        Sets up the UI for displaying graph dimension-related information
        such as diameter, girth, circumference, and vertex-to-vertex distance.
        """
        # Clear previous UI elements
        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Title bar background and label
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white", text="Dimensions", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)

        # Create graph object and compute matrices
        graph = Graph(len(self.vertices), self.edges)
        graph.create_adjacency_matrix()
        graph.create_hashimoto_matrix()

        # Input fields for start and finish vertices
        l3 = Label(self.frame, text="Starting Vertex", bg="grey21", fg="white", font=('Arial 11'))
        l3.grid(row=1, column=0, sticky=N)
        e1 = Entry(self.frame, width=3, font=('Arial 11'))
        e1.grid(row=1, column=1, sticky=NW)
        l4 = Label(self.frame, text="Finishing Vertex", bg="grey21", fg="white", font=('Arial 11'))
        l4.grid(row=1, column=0)
        e2 = Entry(self.frame, width=3, font=('Arial 11'))
        e2.grid(row=1, column=1, sticky=W)

        # Placeholders for displaying results
        l9 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'), width=10)
        l9.grid(row=3, column=1)

        l5 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l5.grid(row=2, column=1, sticky=N)
        l6 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l6.grid(row=2, column=1)
        l7 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l7.grid(row=2, column=1, sticky=S)
        l8 = Label(self.frame, bg="grey21", fg="white", font=('Arial 11'))
        l8.grid(row=3, column=1)

        # Buttons for showing graph measurements
        b1 = Button(self.frame, text="Show Diameter", bg="black", fg="white",
                    font=('Arial 11'), command=lambda:self.show_measurement("D", graph, l5))
        b1.grid(row=2, column=0, sticky=N)

        b2 = Button(self.frame, text="Show Girth", bg="black", fg="white",
                    font=('Arial 11'), command=lambda:self.show_measurement("G", graph, l6))
        b2.grid(row=2, column=0)

        b3 = Button(self.frame, text="Show Circumference", bg="black", fg="white",
                    font=('Arial 11'), command=lambda:self.show_measurement("C", graph, l7))
        b3.grid(row=2, column=0, sticky=S)

        b4 = Button(self.frame, text="Show Distance", bg="black", fg="white",
                    font=('Arial 11'), command=lambda:self.show_distance(graph, l8, e1, e2))
        b4.grid(row=3, column=0)

        # Store references for cleanup
        self.labels = [l1, l2, l3, l4, l5, l6, l7, l8, l9]
        self.buttons = [e1, e2, b1, b2, b3, b4]

        # Add back button
        self.edit_enter_but.config(text="Back", command=self.enter_graph)


    def display_chromatic_colouring(self, graph, l3):
        """
        Displays the chromatic colouring of the given graph on the canvas.
        Updates a label with the chromatic number.
        """
        colours = graph.chromatic_colouring()
        self.reset_colours()

        # Too many colours to display clearly
        if len(set(colours)) > 8:
            l3.config(text="Colouring is too big to show.")
        else:
            l3.config(text="Chromatic Number: " + str(len(set(colours))))
            colour_ops = ["red3", "blue", "green4", "goldenrod", 
                          "magenta3", "SeaGreen3", "dark olive green", "DarkOrange3"]

            # Apply colour to each vertex circle on the canvas
            for v in range(graph.V):
                self.canvas.itemconfigure("circle" + str(v+1), fill=colour_ops[colours[v]-1])


    def colouring_op(self):
        """
        Sets up the UI for graph colouring operations.
        Allows chromatic colouring and manual vertex colouring by user selection.
        """
        self.mode = "colour"

        # Clear previous UI
        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Title bar and heading
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white", text="Colouring", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)

        graph = Graph(len(self.vertices), self.edges)

        # Warning about chromatic colouring limitations
        l4 = Label(self.frame, bg="grey21", fg="white",
                   text="Please note accurate chromatic colourings\n"
                        "can only be guaranteed for graphs\n"
                        "of at most size eight.", font=('Arial 10'))
        l4.grid(row=2, column=0, columnspan=2)

        # Placeholder for chromatic number display
        l3 = Label(self.frame, bg="grey21", fg="white", text="", font=('Arial 12'))
        l3.grid(row=3, column=0, columnspan=2)

        # Button to calculate chromatic colouring
        b1 = Button(self.frame, bg="black", fg="white", text="Chromatic Colouring",
                    font=('Arial 14'), command=lambda:self.display_chromatic_colouring(graph, l3))
        b1.grid(row=2, column=0, columnspan=2, sticky=S)

        # Radiobuttons for manual colouring
        colour_ops = ["red3", "blue", "green4", "goldenrod", 
                      "magenta3", "SeaGreen3", "dark olive green", "DarkOrange3"]
        check_buttons = []
        self.chosen_colour = IntVar()

        # Place first 5 colours in a 2x3 grid
        for i in range(5):
            cb = Radiobutton(self.frame, indicatoron=0, width=3,
                             bg=colour_ops[i], selectcolor=colour_ops[i],
                             variable=self.chosen_colour, value=i+1)
            if 0 <= i <= 1:
                cb.grid(row=1, column=i % 2, sticky=N)
            elif 2 <= i <= 3:
                cb.grid(row=1, column=i % 2)
            else:
                cb.grid(row=1, column=i % 2, sticky=S)
            check_buttons.append(cb)

        # Place remaining 3 colours in a single column
        for i in range(5, 8):
            cb = Radiobutton(self.frame, indicatoron=0, width=3,
                             bg=colour_ops[i], selectcolor=colour_ops[i],
                             variable=self.chosen_colour, value=i+1)
            if i == 5:
                cb.grid(row=1, column=0, columnspan=2, sticky=N)
            elif i == 6:
                cb.grid(row=1, column=0, columnspan=2)
            else:
                cb.grid(row=1, column=0, columnspan=2, sticky=S)
            check_buttons.append(cb)

        # Reset button to clear all colours
        b2 = Button(self.frame, bg="black", fg="white", text="Reset",
                    font=('Arial 10'), command=self.reset_colours)
        b2.grid(row=1, column=1, sticky=S)

        # Store references for cleanup
        self.labels = [l1, l2, l3, l4]
        self.buttons = [b1, b2] + check_buttons

        # Add back button
        self.edit_enter_but.config(text="Back", command=self.enter_graph)


    def planar_info_op(self):
        """
        Sets up the UI to display planar graph information.
        Uses Euler's formula to calculate the number of faces in the graph.
        """
        # Clear previous UI
        self.destroy_all(self.buttons)
        self.destroy_all(self.labels)

        # Title bar and heading
        l1 = Label(self.frame, width=40, bg="grey21")
        l1.grid(row=0, column=0, columnspan=2)
        l2 = Label(self.frame, bg="grey21", fg="white", text="Planar Information", font=('Arial 17 bold'))
        l2.grid(row=0, column=0, columnspan=2)

        # Build graph and find connected components
        graph = Graph(len(self.vertices), self.edges)
        graph.create_adjacency_matrix()
        components = graph.find_connected_components()

        # Collect edges for each component
        comp_edges = []
        for comp in components:
            edgess = []
            for e in graph.edges:
                if (e[0] in comp) and (e[1] in comp):
                    edgess.append(e)
            comp_edges.append(edgess)

        # Use Euler's formula: F = 1 + E - V (per component) + 1 for outside face
        faces = 0
        for i in range(len(components)):
            faces += 1 + len(comp_edges[i]) - len(components[i])
        faces += 1

        # Display number of faces
        l3 = Label(self.frame, bg="grey21", fg="white",
                   text="Number of faces: " + str(faces), font=('Arial 12'))
        l3.grid(row=1, column=0, columnspan=2)

        # Store references for cleanup
        self.labels = [l1, l2, l3]
        self.buttons = []

        # Add back button
        self.edit_enter_but.config(text="Back", command=self.enter_graph)



# Create application
root = Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.title("Simple Graphs Analyser")
root.geometry('%dx%d+%d+%d' % (800, 450, int((screen_width-800)/2), int((screen_height-450)/2)))

app = GraphApp(root)

root.mainloop()
