#%%
import networkx as nx
from networkx.algorithms import matching
from itertools import combinations
from collections import defaultdict
import numpy as np

def find_eulerian_path_or_circuit(edges, start_vertex=0):
    """ 根據給定的邊集尋找歐拉路徑或歐拉迴路。
    Parameters:
        edges (list of tuple): 邊集，每個元組代表一條邊(u, v), 其中u和v是節點。
    Returns:
        list: 歐拉路徑或迴路中的節點序列。如果不存在，返回空列表。
    """

    # 創建無向圖
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    # 尋找起點：奇數度的節點為起點，若無奇數度節點，從任一節點開始
    for vertex, neighbors in graph.items():
        if len(neighbors) % 2 != 0:
            start_vertex = vertex
            break
    # 初始化路徑和堆疊
    path = []
    stack = [start_vertex]
    # 使用 Hierholzer 算法遍歷圖
    while stack:
        vertex = stack[-1]
        if graph[vertex]:
            next_vertex = graph[vertex].pop()
            graph[next_vertex].remove(vertex)   #刪除邊
            stack.append(next_vertex)
        else:
            path.append(stack.pop())
    # 因為是反向構建的，所以返回逆序路徑
    return path[::-1]

def calculate_path_length(path, adj_matrix):
    """ 根據給定的路徑和鄰接矩陣計算總距離。
    Parameters:
        path (list): 經過的節點列表，表示路徑。
        adj_matrix (list of list of int): 鄰接矩陣，其中 adj_matrix[u][v] 表示從節點 u 到節點 v 的距離。
    
    Returns:
        int: 給定路徑的總距離。
    """
    total_length = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        total_length += adj_matrix[u][v]
    return total_length

def find_shortest_paths(num_nodes, adjacency_matrix, source, destination):
    """ 計算從源節點到其他所有節點的最短路徑。
    :param num_nodes: 圖中的節點數量。
    :param adjacency_matrix: 鄰接矩陣，表示節點間的連接和權重。
    :param source: 源節點的索引。
    :param destination: 目的節點的索引。
    :return: 從源節點到目的節點的最短路徑，以節點索引列表形式返回。
    """
    # 初始化所有節點的距離為None，源節點距離為0
    distances = [None] * num_nodes
    distances[source] = 0
    # 待處理節點隊列
    to_visit = [source]
    # 紀錄每個節點的父節點，以重建路徑
    predecessors = [None] * num_nodes
    # 開始進行廣度優先搜索
    while to_visit:
        current = to_visit.pop(0)
        # 檢查所有鄰接節點
        for neighbor in range(num_nodes):
            weight = adjacency_matrix[current][neighbor]
            if weight != 0:  # 存在邊
                # 如果鄰居未訪問或找到更短的路徑
                if distances[neighbor] is None or distances[neighbor] > (distances[current] + weight):
                    distances[neighbor] = distances[current] + weight
                    to_visit.append(neighbor)
                    predecessors[neighbor] = current
    # 從目的節點逆向重建最短路徑
    path = []
    while destination != source:
        path.insert(0, destination)
        destination = predecessors[destination]
    path.insert(0, source)
    
    return path

# 建立奇數度節點之間的完全連通圖
def create_complete_graph(pair_weights, flip_weights=True):
    g = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = -v if flip_weights else v
        g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})
    return g


def find_best_route( distanceMatrix , odd_nodes, edge_list, depot=0):
    # 創建NetworkX圖
    G = nx.Graph()
    for i in range(len(distanceMatrix)):
        for j in range(i + 1, len(distanceMatrix)):
            if distanceMatrix[i][j] != 0:
                G.add_edge(i, j, cost=distanceMatrix[i][j])

    # 計算所有頂點之間的最短路徑
    shortest_paths = dict(nx.floyd_warshall(G, weight='cost'))
    # 構建奇數度頂點之間最短路徑的字典
    odd_node_pairs_shortest_paths = {(u, v): shortest_paths[u][v] for u, v in combinations(odd_nodes, 2)}
    g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)
    matching = nx.algorithms.matching.max_weight_matching(g_odd_complete, maxcardinality=True)
    # 創建最小權重匹配的邊
    # 將匹配結果中的邊根據最短路徑添加到edgelist中
    for u, v in matching:
        path = nx.shortest_path(G, source=u, target=v, weight='cost')
        path_edges = list(zip(path[:-1], path[1:]))
        # print("path_edges:", path_edges)
        for edge in path_edges:
            # print("edge:", edge)
            edge_list.append((edge[0], edge[1]))


    path = find_eulerian_path_or_circuit(edge_list, depot)
    # print("最佳路徑:", path)
    length = calculate_path_length(path, distanceMatrix)
    # print("最佳路徑長度:", length)

    return path, length

def resolution(graph_map):
    temp = graph_map.copy()
    np.fill_diagonal(temp, 0)
    temp[temp != 0] = 10
    odd_nodes = [k for k in range(25) if (temp[k] > 0).sum() % 2 != 0]
    edgeList = []
    for i in range(25):
        for j in range(i + 1, 25):
            if temp[i, j] > 0:
                edgeList.append((i, j))
    distanceMatrix = temp

    path, length = find_best_route( distanceMatrix, odd_nodes, edgeList)
    print("最佳路徑:", path)
    print("最佳長度:", length)
    return length


if __name__ == "__main__":
    from generate_map import graph
    import time
    for i in range(1):
        my_graph = graph(9, True)
        my_graph.cutEdges(1)
        distanceMatrix, _, edgelist = my_graph.getData()
        odd_nodes = my_graph.odd_nodes
        start_time = time.time()
        print(distanceMatrix)
        print(odd_nodes)
        print(edgelist)
        path , length = find_best_route(my_graph, distanceMatrix, odd_nodes, edgelist)
        print("Time:", time.time() - start_time)
        print("最佳路徑:", path)
        print("路徑長度:", length)

    # path, length = traditional_method(my_graph, 0)
    # print("最佳路徑:", path)
    # print("路徑長度:", length)