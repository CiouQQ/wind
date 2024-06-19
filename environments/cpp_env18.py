import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces
import copy
import time
NUM_NODES = 9
Side = int(NUM_NODES**0.5)
num_edges_to_remove = 2
times = 0

def is_connected(matrix):
    graph = nx.from_numpy_array(matrix)
    return nx.is_connected(graph)

class ChinesePostman:
    def __init__(self, num_nodes=NUM_NODES):
        self.num_nodes = num_nodes
        self.map = np.zeros((num_nodes, num_nodes))
        self.current_node = 0
        self.deopt = 0
        self.traveled = np.zeros((num_nodes, num_nodes))
        self.generate_map()
        self.need_to_travel = np.sum(self.map > 0) / 2
        self.traveled_num =  0
        self.max_episode_steps = 512
        self.step_count = 0
        self.renderction = 10
    @property
    def observation_space(self):
        observation_space = spaces.Box(low=0,
                               high=6,
                               shape=(2,9,9), dtype=np.int32)
        print("test")
        return observation_space

    @property
    def action_space(self):
        action_space = spaces.Discrete(self.num_nodes)
        return action_space

    def generate_map(self):
        for i in range(self.num_nodes):
            if i % Side != Side - 1:
                self.map[i][i + 1] = self.map[i + 1][i] = 10
            if i < Side * (Side - 1):
                self.map[i][i + Side] = self.map[i + Side][i] = 10

        edges_removed = 0
        will_removed = 0
        # self.map[5, 4] = self.map[4, 5] = 0
        # self.map[4, 7] = self.map[7, 4] = 0
        # self.map[3, 4] = self.map[4, 3] = 0
        # self.map[4, 1] = self.map[1, 4] = 0
        while will_removed  < num_edges_to_remove:
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
                    break  # 成功移除並保持連通性
                else:
                    self.map[edge_to_remove[0], edge_to_remove[1]] = self.map[edge_to_remove[1], edge_to_remove[0]] = 10
            will_removed += 1
            

    def reset(self):
        self.__init__()
        # pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
        # plot_graph(self.create_networkx_graph(), pos , "test.png")
        self._rewards = []
        self.traveled[self.deopt][self.deopt] = self.traveled[self.deopt][self.deopt] = 1
        combined_map = np.concatenate((self.map, self.traveled), axis=0)
        combined_map = combined_map.reshape(2, 9, 9)
        return combined_map

    def step(self, action):
        done = False
        action = copy.deepcopy(action[0])
        self.renderction = action
        
        self.step_count = self.step_count + 1
        if self.map[self.current_node][action] > 0: #有這個路線
            if self.traveled[self.current_node][action] == 0.0 : #這個路線沒被走過
                reward = 1000
                self.traveled_num += 1
                self.traveled[self.current_node][action] = self.traveled[self.current_node][action] + 1
                self.traveled[action][self.current_node] = self.traveled[self.current_node][action] 
                
            elif self.traveled[self.current_node][action] < 5.0:
                reward = -self.map[self.current_node][action] * self.traveled[self.current_node][action]*10
                self.traveled[self.current_node][action] = self.traveled[self.current_node][action] + 1
                self.traveled[action][self.current_node] = self.traveled[self.current_node][action] 
            else:
                reward = -1000
                done = True
            self.traveled[self.current_node][self.current_node] = self.traveled[self.current_node][self.current_node] = 0
            self.current_node = action
            self.traveled[self.current_node][self.current_node] = self.traveled[self.current_node][self.current_node] = 1
            
        else:                                       #沒有這個路線
            reward = -1000
            done = True
        if self.traveled_num == self.need_to_travel and self.deopt == self.current_node:
            done = True
        if self.step_count >=512:
            done = True


        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
            # print(info)
        else:
            info = None

        combined_map = np.concatenate((self.map, self.traveled), axis=0)
        combined_map = combined_map.reshape(2, 9, 9)
        # print(combined_map)
        return combined_map, reward, done, info

    def create_networkx_graph(self):
        G = nx.Graph()
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.map[i][j] > 0:
                    G.add_edge(i, j, weight=self.map[i][j])
        return G
        
    def render(self):
        # input()
        time.sleep(0.5)
        print(f"Currently at {self.current_node}")
        print("Need to traverse ",self.need_to_travel," edges, already traversed :",self.traveled_num," edges")

    def close(self):
        print("close")

    def plot_graph(self, G, pos, file_name="graph.png"):
        plt.figure(figsize=(15, 15))
        nx.draw(G, pos, with_labels=True, alpha=0.8, font_size=20, node_size=750, edge_color='black')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='c', font_size=20)
        plt.savefig(file_name)
        plt.show()

# Example usage
# cp = ChinesePostman()
# G = cp.create_networkx_graph()

# cp.reset()
# G = cp.create_networkx_graph()
# pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
# cp.plot_graph(G, pos, "test2.png")
# print(cp.action_space)

# for _ in range(50):  # Take 10 random steps in the environment
#     action = int(input("Next will go:"))
#     state, reward, done, info= cp.step(action)
#     # cp.render()
#     # print(f"Action: Move to {action}, Next state: {state[0]}, Reward: {reward}")
#     if done:
#         print("----Done----")
#         cp.reset()
#         G = cp.create_networkx_graph()
#         pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
#         cp.plot_graph(G, pos, "test2.png")

