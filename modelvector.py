import numpy as np
import torch
#test
from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F

from transformer import Transformer
import time

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

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 128, 5, 2,)
            self.conv2 = nn.Conv2d(128, 256, 3, 1, 0)
            self.conv3 = nn.Conv2d(256, 256, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            self.conv1 = nn.Conv1d(655, 128, 5, 2)
            self.conv2 = nn.Conv1d(128, 256, 3, 1)
            self.conv3 = nn.Conv1d(256, 256, 3, 1)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Adjust output size calculation for Conv1d layers
            self.conv_out_size = self.get_conv1d_output(665)
            in_features_next_layer = self.conv_out_size
            # Case: vector observation is available
            # in_features_next_layer = observation_space.shape[0]
        
        # Hidden layer
        self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

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
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
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
        h = obs
        # print("__________________")
        # print(h.shape)
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))
        else:
            batch_size = h.size()[0]  # Reshape to [16, 1, 665]
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            h = h.view(batch_size, -1)  # Flatten the output for further processing

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))
        
        # Forward transformer blocks
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
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
    def get_conv1d_output(self, input_length):
        # Calculate the output length after each Conv1d layer
        # Placeholder calculations: actual values depend on kernel size, stride, and padding
        length = input_length
        length = (length - 5) // 2 + 1  # First Conv1d layer
        length = (length - 3) // 1 + 1  # Second Conv1d layer
        length = (length - 3) // 1 + 1  # Third Conv1d layer
        return length * 256  # Multiply by number of output channels from the last Conv1d layer
    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by layer name
        """
        grads = {}
        # if len(self.observation_space_shape) > 1:
        #     grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3)  
        grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3)  
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
    
    def generate_action_mask(self,obs):
        #  [workers, 2, 9, 9] [workers, node*node+slide*(Side-1)*2] 
        num_actions = 25  # 
        num_workers = obs.shape[0]
        #  [workers, num_actions]
        action_mask = torch.zeros((num_workers, num_actions))
        # 
        for i in range(num_workers):

            matrix = obs[i, :num_actions*num_actions].view(num_actions, num_actions)
            diag_elements = matrix.diagonal()
            rows_to_check = (diag_elements == 1) | (diag_elements == 3)
            rows_to_check = rows_to_check.nonzero(as_tuple=True)[0]
            # 
            # diag_elements = obs[i, 1].diagonal()
            # rows_to_check = (diag_elements == 1).nonzero(as_tuple=True)[0]
            
            # 
            for row in rows_to_check:
                # valid_actions = obs[i, 0, row].nonzero(as_tuple=True)[0]
                valid_actions = matrix[row].nonzero(as_tuple=True)[0]
                valid_actions = valid_actions[valid_actions != row]
                action_mask[i, valid_actions] = 1

            # print(rows_to_check)
            # print(action_mask[i])
        # print("action_mask",action_mask)
        return action_mask