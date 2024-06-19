#%%
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces
import copy
import time
NUM_NODES = 25
Side = int(NUM_NODES**0.5)
# num_edges_to_remove = 2
times = 0

def is_connected(matrix):
    graph = nx.from_numpy_array(matrix)
    return nx.is_connected(graph)

class ChinesePostman:
    def __init__(self, num_nodes=NUM_NODES):
        self.num_nodes = num_nodes
        self.Side = int(np.sqrt(self.num_nodes))
        self.map = np.zeros((num_nodes, num_nodes))
        self.current_node = random.randint(0, 24)
        self.deopt = self.current_node
        self.num_edges_to_remove = random.randint(8, 10)
        self.traveled = np.zeros((num_nodes, num_nodes))
        self.traveled[self.deopt][self.deopt] = self.traveled[self.deopt][self.deopt] = 1
        self.generate_map()
        self.map[self.deopt][self.deopt] =  3
        self.need_to_travel = np.sum(self.map > 0) / 2 - 0.5
        self.traveled_num =  0
        self.max_episode_steps = 64
        self.step_count = 0
        self.renderction = 10
        self.path = []
        self.path.append(self.current_node)  
        self.alledge = 0
        self.target_path = None 
        self.edge_traveled = np.zeros(self.Side * (self.Side - 1) * 2)
        
    @property
    def observation_space(self):
        observation_space = spaces.Box(low=0,
                               high=20,
                               shape=(self.num_nodes * self.num_nodes + 2 * self.Side * (self.Side - 1),), dtype=np.int32)
        print("test")
        return observation_space

    @property
    def action_space(self):
        action_space = spaces.Discrete(self.num_nodes)
        return action_space

    def generate_map(self):
        for i in range(self.num_nodes):
            if i % self.Side != self.Side - 1:
                self.map[i][i + 1] = self.map[i + 1][i] = 10
            if i < self.Side * (self.Side - 1):
                self.map[i][i + self.Side] = self.map[i + self.Side][i] = 10

        edges_removed = 0
        will_removed = 0
        # self.map[7, 8] = self.map[8, 7] = 0
        # self.map[4, 1] = self.map[1, 4] = 0
        # self.map[7, 4] = self.map[4, 7] = 0
        # self.map[1, 2] = self.map[2, 1] = 0
        while will_removed  < self.num_edges_to_remove:
            existing_edges = np.argwhere(self.map > 0)
            # print("Wii deal ",will_removed,"\n__\n")
            if not existing_edges.size:
                # print("No more edges to remove.")
                break

            for _ in range(1*len(existing_edges)):
                edge_to_remove = existing_edges[np.random.randint(0, len(existing_edges))]
                self.map[edge_to_remove[0], edge_to_remove[1]] = self.map[edge_to_remove[1], edge_to_remove[0]] = 0
                if is_connected(self.map):
                    # print("The number is ",edges_removed," The node is ",edge_to_remove,"\n__\n")
                    edges_removed += 1
                    break  # 
                else:
                    self.map[edge_to_remove[0], edge_to_remove[1]] = self.map[edge_to_remove[1], edge_to_remove[0]] = 10
            will_removed += 1
            

    def reset(self):
        self.__init__()
        # pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
        # plot_graph(self.create_networkx_graph(), pos , "test.png")
        self._rewards = []
        flattened_map = self.map.flatten()
        combined_vector = np.concatenate((flattened_map, self.edge_traveled))
        # print(combined_vector)
        return combined_vector

    def step(self, action):
        done = False
        reward = 0
        action = copy.deepcopy(action[0])
        self.renderction = action
        
        self.step_count = self.step_count + 1
        if self.map[self.current_node][action] > 0: #
            if self.traveled[self.current_node][action] == 0.0 : #
                reward = reward + 10
                self.traveled_num += 1
                self.traveled[self.current_node][action] = self.traveled[self.current_node][action] + 1
                self.traveled[action][self.current_node] = self.traveled[self.current_node][action] 
                self.update_edge_traveled(self.current_node, action)
                
            elif self.traveled[self.current_node][action] < 5.0:
                reward = reward - ((self.map[self.current_node][action] * self.traveled[self.current_node][action])**2)/100
                self.traveled[self.current_node][action] = self.traveled[self.current_node][action] + 1
                self.traveled[action][self.current_node] = self.traveled[self.current_node][action] 
                self.update_edge_traveled(self.current_node, action)
            else:
                reward = reward-10
                # with open('test.txt', 'a') as file:
                #     file.write('*')
                done = True
            self.traveled[self.current_node][self.current_node] = self.traveled[self.current_node][self.current_node] = 0
            self.map[self.current_node][self.current_node] = self.map[self.current_node][self.current_node] - 1
            self.current_node = action
            self.map[self.current_node][self.current_node] = self.map[self.current_node][self.current_node] + 1
            self.traveled[self.current_node][self.current_node] = self.traveled[self.current_node][self.current_node] = 1
            
        else:                                     #
            reward = reward-100
            with open('test.txt', 'a') as file:
                file.write('9')
            done = True

        if self.traveled_num == self.need_to_travel and self.alledge == 0:
            with open('test.txt', 'a') as file:
                file.write('1')
            self.alledge = 1
            # reward = reward + 100
        if self.alledge == 1 and self.deopt == self.current_node:
            with open('test.txt', 'a') as file:
                file.write('2')
            reward = reward + 200
            done = True

        if self.step_count >=64:
            with open('test.txt', 'a') as file:
                file.write('over')
            done = True


        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
            # print(info)
        else:
            info = None

        # combined_map = np.concatenate((self.map, self.traveled), axis=0)
        # combined_map = combined_map.reshape(2, NUM_NODES, NUM_NODES)
        # print(combined_map)
        self.path.append(self.current_node)
        flattened_map = self.map.flatten()
        combined_vector = np.concatenate((flattened_map, self.edge_traveled))
        # print(combined_vector)
        return combined_vector, reward, done, info

    def create_networkx_graph(self):
        G = nx.Graph()
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.map[i][j] > 0:
                    G.add_edge(i, j, weight=self.map[i][j])
        return G
    def get_pos(self):
        pos = {i: (i % self.Side, self.Side - 1 - i // self.Side) for i in range(NUM_NODES)}
        return pos
        
    def render(self):
        # input()
        # time.sleep(0.5)
        # print("Current path:", self.path)
        print("edge_traveled",self.map)
        # print(f"Currently at {self.current_node}")
        # print("Need to traverse ",self.need_to_travel," edges, already traversed :",self.traveled_num," edges")

    def close(self):
        print("close")

    def plot_graph(self, G, pos, file_name="graph.png"):
        plt.figure(figsize=(15, 15))
        nx.draw(G, pos, with_labels=True, alpha=0.8, font_size=20, node_size=750, edge_color='black')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='c', font_size=20)
        plt.savefig(file_name)
        plt.show()

    def find_untraveled_edges(self):
        untraveled_edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):  # 
                if self.map[i][j] > 0 and self.traveled[i][j] == 0:
                    untraveled_edges.append((i, j))
        return untraveled_edges
    def print_shortest_path(self, target):
        G = self.create_networkx_graph()  # 
        if target >= self.num_nodes or target < 0:
            return f"Invalid target node: {target}. Node must be between 0 and {self.num_nodes - 1}"
        
        if self.current_node == target:
            return f"The current node {self.current_node} is the target node."

        try:
            # 
            path = nx.shortest_path(G, source=self.current_node, target=target)
            if len(path) > 1:
                return path[1:]
            else:
                return path
        except nx.NetworkXNoPath:
            # 
            return f"No path from node {self.current_node} to node {target}."
        
    def print_path_to_random_untraveled_edge(self):
        # 
        untraveled_edges = self.find_untraveled_edges()
        if not untraveled_edges:
            return self.print_shortest_path(0)

        # 
        
        random_edge = random.choice(untraveled_edges)
        target_node = random.choice(random_edge)  # 
        if self.current_node == target_node:
            target_node = random_edge[0] if target_node == random_edge[1] else random_edge[1]

        other_node = random_edge[0] if target_node == random_edge[1] else random_edge[1]
        path = self.print_shortest_path(target_node)
        # print(path,"--------------")
        if other_node not in path:
            path.append(other_node)
        # print("The target is :", target_node)
        # 
        return path
    def manage_path_to_untraveled_edge(self):
        if self.target_path is None or not self.target_path:
            # 
            path = self.print_path_to_random_untraveled_edge()
            if isinstance(path, list):  # 
                self.target_path = path
            return path
        else:
            # 
            if self.current_node == self.target_path[0]:
                self.target_path.pop(0)  # 
            if not self.target_path:
                path = self.print_path_to_random_untraveled_edge()
                if isinstance(path, list):  # 
                    self.target_path = path
            return self.target_path
    def update_edge_traveled(self, from_node, to_node):
        
        from_node, to_node = min(from_node, to_node), max(from_node, to_node)

        if from_node % self.Side == to_node % self.Side:  # 同一列
            col_index = from_node % self.Side  #xx
            row_distance = from_node // self.Side #y
            index = (self.Side - 1)*col_index + row_distance #2x+y
        else:  # 同一行
            row_index = from_node // self.Side #x
            col_distance = from_node % self.Side  #y
            index = (self.Side - 1)*row_index + col_distance+ self.Side * (self.Side - 1)  #2x+y+side*(side-1)
        
       
        self.edge_traveled[index] += 1  
        
# Example usage
# Side = NUM_NODES**(1/2)
# cp = ChinesePostman()
# cp.reset()
# G = cp.create_networkx_graph()
# pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
# cp.plot_graph(G, pos, "test2.png")
# print(cp.action_space)
# cp.render()
# for _ in range(50):  # Take 10 random steps in the environment
#     action = int(input("Next will go:"))
#     state, reward, done, info= cp.step(action)
#     cp.render()
#     # print(cp.edge_traveled)
#     # print("path:", cp.manage_path_to_untraveled_edge()) 
#     # cp.render()
#     # print(f"Action: Move to {action}, Next state: {state[0]}, Reward: {reward}")
#     if done:
#         print("----Done----")
#         cp.reset()
#         cp.render()
#         G = cp.create_networkx_graph()
#         pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
#         cp.plot_graph(G, pos, "test2.png")