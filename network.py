# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "networkx",
#     "scipy",
# ]
# ///
"""
Inspired by https://blog.otoro.net/2015/07/31/neurogram/
"""
import math
import random
import matplotlib.pyplot as plt
import networkx as nx  # for drawing the graphs

innovation = 1
node_number = 1

def linear(x):
    return x

def sigmoid(x):
    # Clamp x to prevent overflow in exp function
    x = max(-500, min(500, x))
    return 1/(1+math.exp(-x))

def sine(x):
    return math.sin(x/50)

def cosine(x):
    return math.cos(x/50)

def relu(x):
    return max(0, x)



activations = [linear, sigmoid, sine, cosine, relu]

class Node:
    def __init__(self, name, activation):
        self.name = name
        self.value = 0
        self.activation = activation

    def __repr__(self):
        return f"Node(name={self.name}, value={self.value}, activation={self.activation})"

    def reset(self):
        self.value = 0

    def add(self, other):
        self.value += other

    def copy(self):
        return Node(self.name, self.activation)

    def __call__(self):
        return self.activation(self.value)

class Edge:
    def __init__(self, from_node, to_node, innov=None, weight=None, enabled=True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight if weight else random.uniform(-2, 2)
        self.enabled = enabled

        if innov:
            self.innovation = innov
        else:
            global innovation
            self.innovation = innovation
            innovation += 1

    def __repr__(self):
        return f"Edge(({self.from_node}, {self.to_node}), weight={self.weight}, innovation={self.innovation}, enabled={self.enabled})"

    def forward(self):
        self.to_node.add(self.from_node() * self.weight)

class Network:
    def __init__(self, input_nodes: list[Node], hidden_nodes: list[Node], output_nodes: list[Node], edges: list[Edge]):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.edges = edges

    def topological_sort(self):
        """Perform a topological sort of the nodes by depth-first search"""
        # Depth-first search
        s = [node for node in self.input_nodes]
        nodes_visited = []  # keep track of the order in which nodes are visited
        while len(s) > 0:
            v = s.pop()
            if v not in nodes_visited:
                nodes_visited.append(v)
                for edge in self.edges:
                    if edge.from_node == v:
                        s.append(edge.to_node)

        # Reverse postordering of visited nodes produces a topological sorting
        nodes_ordered = []
        for node in nodes_visited[::-1]:
            if node not in nodes_ordered:
                nodes_ordered.append(node)
        return nodes_ordered[::-1]

    def forward(self, input):
        # Clear values of all nodes
        for node in self.input_nodes + self.hidden_nodes + self.output_nodes:
            node.reset()

        # Fill the values of the input nodes
        for val, node in zip(input, self.input_nodes):
            node.add(val)

        # Process the rest of the nodes in the right order
        nodes_sorted = self.topological_sort()
        for node in nodes_sorted:
            if node in self.input_nodes:
                continue
            for edge in self.edges:
                if edge.to_node == node:
                    edge.forward()

        return [output_node() for output_node in self.output_nodes]

    def mutate(self):
        network = self.copy()

        if random.random() < 0.5:
            network.add_edge()
        if random.random() < 0.5:
            network.add_node()
        if random.random() < 0.8:
            network.change_weights()
        if random.random() < 0.3:
            network.change_activation()

        return network

    def add_edge(self):
        """Add an edge between two random nodes"""
        from_node = random.sample(self.input_nodes + self.hidden_nodes, 1)[0]
        to_node = random.sample(self.hidden_nodes + self.output_nodes, 1)[0]

        if from_node == to_node:
            return

        new_edge = Edge(from_node, to_node)
        self.edges.append(new_edge)

    def add_node(self):
        """Create a new node and put it inside an existing edge"""
        global activations, node_number
        activation = random.sample(activations, 1)[0]
        new_node = Node(f"h{node_number}", activation)
        node_number += 1
        self.hidden_nodes.append(new_node)

        random.shuffle(self.edges)
        old_edge = self.edges.pop()
        new_edge1 = Edge(old_edge.from_node, new_node)
        new_edge2 = Edge(new_node, old_edge.to_node)
        self.edges += [new_edge1, new_edge2]

    def change_weights(self):
        edge = random.sample(self.edges, 1)[0]
        edge.weight = (random.random() - 0.5)*2

    def change_activation(self):
        node = random.sample(self.hidden_nodes, 1)[0]
        node.activation = random.sample(activations, 1)[0]

    def copy(self):
        input_nodes = [node.copy() for node in self.input_nodes]
        hidden_nodes = [node.copy() for node in self.hidden_nodes]
        output_nodes = [node.copy() for node in self.output_nodes]
        all_nodes = {node.name: node for node in input_nodes + output_nodes + hidden_nodes}
        edges = [Edge(all_nodes[edge.from_node.name], all_nodes[edge.to_node.name], edge.innovation, edge.weight) for edge in self.edges]

        network = Network(input_nodes, hidden_nodes, output_nodes, edges)

        return network

    def __repr__(self):
        return ",".join(str(node) for node in self.hidden_nodes)


def crossover(network1: Network, network2: Network) -> Network:
    global node_number
    edges1, edges2 = network1.edges, network2.edges

    # Align the edges by innovation number
    edges1 = {edge.innovation: edge for edge in edges1}
    edges2 = {edge.innovation: edge for edge in edges2}

    input_nodes = [node.copy() for node in network1.input_nodes]
    output_nodes = [node.copy() for node in network1.output_nodes]
    hidden_nodes = [Node(f"h{i+1}", linear) for i in range(node_number)]  # activations will be set correctly later
    all_nodes = {node.name: node for node in input_nodes + output_nodes + hidden_nodes}

    # Keep track of which edges to add
    to_add_edges = []
    # Add edges from both parents
    for innovation in set(edges1.keys()) | set(edges2.keys()):
        edge1 = edges1.get(innovation)
        edge2 = edges2.get(innovation)
        if edge1 and edge2:
            # Coin flip if both have this edge
            if random.random() < 0.5:
                to_add_edges.append(edge1)
            else:
                to_add_edges.append(edge2)
        # If only network1 has an edge, add it
        elif edge1:
            to_add_edges.append(edge1)
        # If only network2 has an edge, add it
        else:
            to_add_edges.append(edge2)
    # Copy edges while pointing to new nodes
    edges = []
    used_hidden_nodes = []
    for edge in to_add_edges:
        new_edge = Edge(all_nodes[edge.from_node.name], all_nodes[edge.to_node.name], edge.innovation, edge.weight)
        edges.append(new_edge)
        # Set activation of hidden node correctly
        if new_edge.to_node in hidden_nodes:
            new_edge.to_node.activation = edge.to_node.activation
            used_hidden_nodes.append(new_edge.to_node)
        if new_edge.from_node in hidden_nodes:
            used_hidden_nodes.append(new_edge.from_node)

    network = Network(input_nodes, used_hidden_nodes, output_nodes, edges)

    return network

def random_network():
    x, y, d, bias = Node("x", linear), Node("y", linear), Node("d", linear), Node("bias", linear)
    global node_number
    global activations
    activation = random.sample(activations, 1)[0]
    h1 = Node(f"h{node_number}", activation)
    node_number += 1
    r, g, b = Node("r", sigmoid), Node("g", sigmoid), Node("b", sigmoid)

    xh1, yh1, dh1, biash1 = Edge(x, h1), Edge(y, h1), Edge(d, h1), Edge(bias, h1)
    h1r, h1g, h1b = Edge(h1, r), Edge(h1, g), Edge(h1, b)

    network = Network([x, y, d, bias], [h1], [r, g, b], [xh1, yh1, dh1, biash1, h1r, h1g, h1b])

    return network.mutate().mutate().mutate()


def show_networks(networks: list[Network]):
    f, axes = plt.subplots(ncols=3, nrows=math.ceil(len(networks)/3)*2)
    for i in range(len(networks)):
        ax = axes[2*(i // 3), i % 3]
        image = [[[] for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for x in range(WIDTH):
            for y in range(HEIGHT):
                d = math.sqrt(((x-WIDTH//2)**2 + (y-HEIGHT//2)**2))
                image[y][x] = networks[i].forward([x, y, d, 1])
        ax.set_title(i)
        ax.imshow(image)

        # Generate and plot a Networkx object
        ax = axes[2*(i // 3) + 1, i % 3]
        g = nx.DiGraph()
        for e in networks[i].edges:
            g.add_edge(e.from_node.name, e.to_node.name)
        for n in networks[i].input_nodes:
            g.add_node(n.name, column="0")
        for n in networks[i].hidden_nodes:
            g.add_node(n.name, column="1")
        for n in networks[i].output_nodes:
            g.add_node(n.name, column="2")
        pos = nx.multipartite_layout(g, subset_key="column")
        nx.draw_networkx(g, pos=pos, ax=ax, node_size=10)

    plt.show()


N_NETWORKS = 6
WIDTH, HEIGHT = 128, 128

networks = [random_network() for _ in range(N_NETWORKS)]

while True:
    show_networks(networks)
    selection = input("Images to combine (e.g. '0,2'): ")
    im1, im2 = selection.split(",")
    n1, n2 = networks[int(im1)], networks[int(im2)]
    networks = [crossover(n1, n2), n1.mutate(), n2.mutate(), n1.mutate(), n2.mutate(), n1.mutate()]
