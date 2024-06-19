from collections import defaultdict
import heapq
import graph
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import os
import time
from datetime import datetime

def dijkstra_shortest_path(graph, start, end):
    dist = {vertex: float('infinity') for vertex in graph}
    dist[start] = 0
    pq = [(0, start)]
    prev = {vertex: None for vertex in graph}

    while pq:
        current_dist, vertex = heapq.heappop(pq)
        if vertex == end:
            break
        for neighbor in graph[vertex]:
            distance = current_dist + 1
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = vertex
                heapq.heappush(pq, (distance, neighbor))

    path = []


    if prev[end] is not None or end == start:
        while end:
            path.insert(0, end)
            end = prev[end]
    return path

def find_shortest_paths_combination(odd_vertices, edges):
    graph = create_adj_list(edges)
    shortest_combination = []

    while odd_vertices:
        shortest_path_length = float('infinity')
        shortest_path = []
        u = odd_vertices[0]

        for v in odd_vertices[1:]:
            path = dijkstra_shortest_path(graph, u, v)
            if 1 < len(path) < shortest_path_length:
                shortest_path_length = len(path)
                shortest_path = path

        for i in range(len(shortest_path) - 1):
            shortest_combination.append((shortest_path[i], shortest_path[i+1]))

        odd_vertices.remove(u)
        odd_vertices.remove(shortest_path[-1])

    return shortest_combination


def extract_edges_from_gv1_graph():
    G = graph.get_graph()  
    edges = []
    for u, v in G.edges():
        edges.append((u, v))  
    return edges

def create_adj_list(edges):
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_list

def remove_edge(adj_list, u, v):
    adj_list[u].remove(v)
    adj_list[v].remove(u)

def make_eulerian_if_needed(edges):
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    odd_vertices = [v for v, d in degree.items() if d % 2 != 0]
    
    print("\nodd_ver:",odd_vertices)
    shortest_paths_combination = find_shortest_paths_combination(odd_vertices, edges)

    integrated_edges = edges + shortest_paths_combination

    
    return shortest_paths_combination, integrated_edges

def find_euler_path(edges, start):
    adj_list = create_adj_list(edges)
    stack = [start]
    path = []

    while stack:
        vertex = stack[-1]
        if adj_list[vertex]:
            next_vertex = adj_list[vertex][0]
            stack.append(next_vertex)
            remove_edge(adj_list, vertex, next_vertex)
        else:
            path.append(stack.pop())
    
    edge_path = []
    for i in range(1, len(path)):
        if (path[i-1], path[i]) in edges or (path[i], path[i-1]) in edges:
            edge_path.append((path[i-1], path[i]))
    return edge_path



def plot_graph_with_path(G, pos, euler_path, repeat_path, step, result):
    plt.figure(figsize=(15.0, 15.0))
    nx.draw(G, pos, with_labels=True, alpha=0.8, font_size=20, node_size=750, edge_color='black', connectionstyle='arc3, rad=0.2')
    

    edge_labels = nx.get_edge_attributes(G, 'weight')
    edges = list(G.edges)
    for (i, j), label in edge_labels.items():
        (x1, y1) = pos[i]
        (x2, y2) = pos[j]
        (x, y) = ((x1 + x2) / 2, (y1 + y2) / 2)

                # 根據弧度計算偏移量
        if (i, j) in edges:
            rad = 0.15
        else:
            rad = -0.15
        dx, dy = (y2 - y1) * rad, (x1 - x2) * rad

                # 繪製邊標籤
        plt.text(x + dx, y + dy, s=label, fontsize=20, color='c', horizontalalignment='center', verticalalignment='center')

    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='c', font_size=20)
    edge_traversal_counts = {}  # 檢查每條edge重複次數
    

    for i, (u, v) in enumerate(euler_path[:step+1]):
        current_node = int(v)
        edge = tuple(sorted((u, v)))
        edge_traversal_counts[edge] = edge_traversal_counts.get(edge, 0) + 1  

        if edge_traversal_counts[edge] == 1:
            color = 'red'    
        elif edge_traversal_counts[edge] == 2:
            color = 'green'  
        elif edge_traversal_counts[edge] == 3:
            color = 'blue'  
        else:
            color = 'purple' 

        if (u, v) in edges:
            rad = -0.2
        else:
            rad = 0.2

        nx.draw_networkx_edges(G, pos, edgelist=[(int(u), int(v))], edge_color=color, width=2, connectionstyle=f'arc3, rad={rad}')
    nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='Orange', node_size=500)
    plt.axis('off')
    file_name = f"verify/{result}/tmp_path_step_{step}.png"
    plt.savefig(file_name)
    plt.close()
    return file_name

def generate_gif_from_path(G, pos, path, repeat_path, result, distance):
    filenames = []
    # 畫每步
    print("path",path)
    for step in range(len(path)):
        filename = plot_graph_with_path(G, pos, path, repeat_path, step, result)
        filenames.append(filename)
    
    # GIF
    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d_%H%M%S")  
    filename = f"verify/{result}/{distance}path_animation_{formatted_now}.gif"
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(filename, images, fps=3, loop=3)  
    with open(f"verify/{result}/verify.txt", 'a') as file:
            file.write(f'\n{formatted_now}:{path}') 
    for filename in filenames:
        os.remove(filename)


# if __name__ == "__main__":

   
#     edges = extract_edges_from_gv1_graph()  #抓地圖
#     edges = [(str(u), str(v)) for u, v in edges]
#     print(edges,"\n")
#     start_node = input("Input the depot: ")
#     start_time = time.time()
#     repeat_path, edges = make_eulerian_if_needed(edges)  # 將圖變成euler迴圈，完整的cpp
    
#     euler_path = find_euler_path(edges, start_node)   #找出euler路徑
    

#     euler_path = [(min(u, v), max(u, v)) for u, v in euler_path]  
#     repeat_path = [(min(u, v), max(u, v)) for u, v in repeat_path]
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print("time is ",execution_time)
#     #-----------
#     # path = []
#     # path.append(('0', '3'))
#     # path.append(('3', '4'))
#     # path.append(('4', '7'))
#     # path.append(('7', '6'))
#     # path.append(('6', '3'))
#     # path.append(('3', '4'))
#     # path.append(('4', '1'))
#     # path.append(('1', '2'))
#     # path.append(('2', '5'))
#     print(euler_path)
#     # generate_gif_from_path(graph.get_graph() , graph.get_pos(), path, repeat_path)
#     #------------
#     generate_gif_from_path(graph.get_graph() , graph.get_pos(), euler_path, repeat_path)
    

