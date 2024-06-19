import networkx as nx
import matplotlib.pyplot as plt
import random

def adjust_positions(pos, edges):
    for edge in edges:
        start, end, distance = int(edge[0]), int(edge[1]), edge[2]['distance']
        dx, dy = pos[end][0] - pos[start][0], pos[end][1] - pos[start][1]
        z = (dx**2 + dy**2) ** 0.5
        factor = distance / 50  
        pos[end] = (pos[start][0] + factor * dx / z, pos[start][1] + factor * dy / z)
    return pos

def generate_grid_graph(node_number):
    G = nx.Graph()
    for i in range(node_number**2 ):
        print(i)
        if i % node_number != node_number - 1:
            G.add_edge(i, i + 1, distance=50)
        if i < node_number * (node_number - 1):
            G.add_edge(i, i + node_number, distance=50)
    #-------
    # G.remove_edge(0, 1)
    # G.remove_edge(5, 8)
    #-------
    return G

def remove_edges_with_constraints(G, num_edges_to_remove):
    edges_removed = 0
    while edges_removed < num_edges_to_remove:
        edge = random.choice(list(G.edges))
        if all(len(list(G.neighbors(n))) > 2 for n in edge):
            G.remove_edge(*edge)
            edges_removed += 1

def plot_graph(G, pos, file_name="graph.png"):
    plt.figure(figsize=(15, 15))
    nx.draw(G, pos, with_labels=True, alpha=0.8, font_size=20, node_size=750, edge_color='black')
    labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='c', font_size=20)
    if file_name:
        plt.savefig(file_name)
    plt.show()

def get_graph():
    return G

def get_pos():
    return pos

# Node_number = 3
# num_edges_to_remove = 1

# G_full = generate_grid_graph(Node_number)
# pos = {i: (i % Node_number, Node_number - 1 - i  // Node_number) for i in range(0, Node_number**2 )}
# G = G_full.copy()
# remove_edges_with_constraints(G, num_edges_to_remove)
# # adjust_positions(pos, G.edges(data=True))
# plot_graph(G, pos, "final_graph.png")
