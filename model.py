import numpy as np
import torch
#test
from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F

from transformer import Transformer
import time
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GraphModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(GraphModule, self).__init__()
        self.conv1 = GCNConv(input_channels, 8)
        self.conv2 = GCNConv(8, output_channels)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    
class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_length):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.memory_layer_size = config["transformer"]["embed_dim"]
        self.observation_space_shape = observation_space.shape
        self.max_episode_length = max_episode_length
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # Observation encoder
        # if len(self.observation_space_shape) > 1:
        #     # Case: visual observation is available
        #     # Visual encoder made of 3 convolutional layers
        #     self.conv1 = nn.Conv2d(observation_space.shape[0], 128, 5, 2,)
        #     self.conv2 = nn.Conv2d(128, 256, 3, 1, 0)
        #     self.conv3 = nn.Conv2d(256, 256, 3, 1, 0)
        #     nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
        #     nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        #     nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
        #     # Compute output size of convolutional layers
        #     self.conv_out_size = self.get_conv_output(observation_space.shape)
        #     in_features_next_layer = self.conv_out_size
        # else:
        #     # Case: vector observation is available
        #     in_features_next_layer = observation_space.shape[0]
        # Hidden layer
        self.lin_hidden = nn.Linear(7500, self.memory_layer_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # self.graph_module = GraphModule(1, 16)  # Assuming 1 feature per node
        self.conv1 = GCNConv(50, 75)
        self.conv2 = GCNConv(75, 150)
        self.conv3 = GCNConv(150, 150)
        self.conv4 = GCNConv(150, 300)
        self.conv5 = GCNConv(300, 300)
        # Transformer Blocks
        self.transformer = Transformer(config["transformer"], self.memory_layer_size, self.max_episode_length)

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(2))
            self.policy_branches.append(actor_branch)
            
        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, memory:torch.tensor, memory_mask:torch.tensor, memory_indices:torch.tensor):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            memory {torch.tensor} -- Episodic memory window
            memory_mask {torch.tensor} -- Mask to prevent the model from attending to the padding
            memory_indices {torch.tensor} -- Indices to select the positional encoding that matches the memory window

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value function: Value
        """
        # Set observation as input to the model
        # h = obs
        # # Forward observation encoder
        # if len(self.observation_space_shape) > 1:
        #     batch_size = h.size()[0]
        #     # Propagate input through the visual encoder
        #     h = self.leaky_relu(self.conv1(h))
        #     h = self.leaky_relu(self.conv2(h))
        #     h = self.leaky_relu(self.conv3(h))
        #     # Flatten the output of the convolutional layers
        #     h = h.reshape((batch_size, -1))

        # # Feed hidden layer
        # h = self.leaky_relu(self.lin_hidden(h))
        x1, x2, edge_index = self.process_observation(obs)
        x = torch.cat((x1, x2), dim=2)
        # print("x",x.shape) # torch.Size([8, 625, 2])
        # print("obs",obs.shape) #torch.Size([8, 2, 25, 25])   ([8, 25, 50])
        # h = self.graph_module(x, edge_index)
        h = F.leaky_relu(self.conv1(x, edge_index))
        h = F.leaky_relu(self.conv2(h, edge_index))
        h = F.leaky_relu(self.conv3(h, edge_index))
        h = F.leaky_relu(self.conv4(h, edge_index))
        h = F.leaky_relu(self.conv5(h, edge_index))
        h = h.reshape((obs.size()[0], -1))
        h = self.leaky_relu(self.lin_hidden(h))
        # Forward transformer blocks
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = self.leaky_relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = self.leaky_relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        logits = [branch(h_policy) for branch in self.policy_branches]
        action_mask = self.generate_action_mask(obs)
        if action_mask is not None:
            for i in range(len(logits)):
                logits[i] = logits[i] + action_mask.log()

        # Head: Policy
        pi = [Categorical(logits=logit) for logit in logits]
        # pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]
        
        return pi, value, memory
    def process_observation(self, obs):
        # Placeholder: you need to convert your obs to x and edge_index
        # This is a simplified example assuming obs directly gives the features and edges
        x1 = obs[:, 0, :, :].view(obs.size(0), -1, 25)  # Reshape to (batch_size, num_nodes, num_features) torch.Size([8, 625, 1])
        x2 = obs[:, 1, :, :].view(obs.size(0), -1, 25)
        edge_index = self.create_edge_index(5)  # Assuming 5x5 grid
        return x1, x2, edge_index
    
    def create_edge_index(self, n):
        # Creates edge index for an n x n grid
        idx = torch.arange(n*n).view(n, n)
        edge_index = []
        for i in range(n):
            for j in range(n):
                if i < n-1:
                    edge_index.append([idx[i, j], idx[i+1, j]])
                    edge_index.append([idx[i+1, j],idx[i, j] ])
                if j < n-1:
                    edge_index.append([idx[i, j], idx[i, j+1]])
                    edge_index.append([idx[i, j+1],idx[i, j] ])
        return torch.tensor(edge_index).t().contiguous()
    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by layer name
        """
        grads = {}
        if len(self.observation_space_shape) > 1:
            grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)    
            
        grads["linear_layer"] = self._calc_grad_norm(self.lin_hidden)
        
        transfomer_blocks = self.transformer.transformer_blocks
        for i, block in enumerate(transfomer_blocks):
            grads["transformer_block_" + str(i)] = self._calc_grad_norm(block)
        
        for i, head in enumerate(self.policy_branches):
            grads["policy_head_" + str(i)] = self._calc_grad_norm(head)
        
        grads["lin_policy"] = self._calc_grad_norm(self.lin_policy)
        grads["value"] = self._calc_grad_norm(self.lin_value, self.value)
        grads["model"] = self._calc_grad_norm(self, self.value)
          
        return grads
    
    def _calc_grad_norm(self, *modules):
        """Computes the norm of the gradients of the given modules.

        Arguments:
            modules {list} -- List of modules to compute the norm of the gradients of.

        Returns:
            {float} -- Norm of the gradients of the given modules. 
        """
        grads = []
        for module in modules:
            for name, parameter in module.named_parameters():
                grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None
    
    # def generate_action_mask(self,obs):
    #     #  [workers, 2, 9, 9] [workers, node*node+slide*(Side-1)*2] 
    #     num_actions = 25  # 
    #     num_workers = obs.shape[0]
    #     #  [workers, num_actions]
    #     action_mask = torch.zeros((num_workers, num_actions))
    #     # 
    #     for i in range(num_workers):
    #         # 
    #         diag_elements = obs[i, 1].diagonal()
    #         rows_to_check = (diag_elements == 1).nonzero(as_tuple=True)[0]
            
    #         # 
    #         for row in rows_to_check:
    #             valid_actions = (obs[i, 0, row] > 0).nonzero(as_tuple=True)[0]
    #             valid_actions = valid_actions[valid_actions != row]
    #             action_mask[i, valid_actions] = 1
    #     # print("action_mask",action_mask)
    #     return action_mask

    def generate_action_mask(self, obs):
        num_workers = obs.shape[0]
        side = int(np.sqrt(obs.shape[2]))  # 
        num_actions = (side - 1) * side * 2  # 
        action_mask = torch.ones((num_workers, num_actions))  # 

        for i in range(num_workers):
            current_position = obs[i, 1].diagonal().nonzero()[0]  # 
            if len(current_position) == 0:
                continue  # 

            # 
            for action in range(num_actions):
                node1, node2 = self.decode_edge_action(action, side)
                # 
                if obs[i, 0, node1, node2] == 0 or obs[i, 1, node1, node2] > 0:
                    action_mask[i, action] = 0  # 

        # 憒���𨅯�券�典𢆡雿𣈯�質◤�桃蔗嚗�鍳�鍂銝𦒘�枏�梶㮾�餌�颲�
        for i in range(num_workers):
            if torch.all(action_mask[i] == 0):  # 璉��䰻�糓�炏����匧𢆡雿𣈯�質◤�桃蔗
                for j in current_position:
                    for action in range(num_actions):
                        node1, node2 = self.decode_edge_action(action, side)
                        if j in [node1, node2] and obs[i, 0, node1, node2] > 0:
                            action_mask[i, action] = 1  # �鍳�鍂銝𦒘�枏�梶㮾�餌�颲�
                            break
        # print("actionmask",action_mask)
        return action_mask

    def decode_edge_action(self, action, side):
        if action < side * (side - 1):  # 瘞游像颲�
            row = action // (side - 1)
            col = action % (side - 1)
            node1 = row * side + col
            node2 = node1 + 1
        else:  # ���凒颲�
            action -= side * (side - 1)
            row = action // side
            col = action % side
            node1 = row * side + col
            node2 = node1 + side
        return node1, node2