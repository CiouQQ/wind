#%%
import numpy as np
import pickle
import torch
import tkinter as tk
from cpp import generate_gif_from_path
from tkinter import filedialog
from docopt import docopt
from model import ActorCriticModel
from utils import create_env
import time
from datetime import datetime


def init_transformer_memory(trxl_conf, max_episode_steps, device):
    """Returns initial tensors for the episodic memory of the transformer.

    Arguments:
        trxl_conf {dict} -- Transformer configuration dictionary
        max_episode_steps {int} -- Maximum number of steps per episode
        device {torch.device} -- Target device for the tensors

    Returns:
        memory {torch.Tensor}, memory_mask {torch.Tensor}, memory_indices {torch.Tensor} -- Initial episodic memory, episodic memory mask, and sliding memory window indices
    """
    # Episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
    # Episdic memory tensor
    memory = torch.zeros((1, max_episode_steps, trxl_conf["num_blocks"], trxl_conf["embed_dim"])).to(device)
    # Setup sliding memory window indices
    repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0), trxl_conf["memory_length"] - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in range(max_episode_steps - trxl_conf["memory_length"] + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices
def load_model_path(default = False):
    root = tk.Tk()
    if default:
        return f"models/20240420-232428/2.391.nn "
    else:
        #default folder is model
        file_path = filedialog.askopenfilename(initialdir = "models")
        root.destroy()
        return file_path

def main(model,times):
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
    """
    # options = docopt(_USAGE)
    # model_path = options["--model"]
    # model_path = load_model_path()   
    # Set inference device and default tensor type
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Instantiate environment
    env = create_env(config["environment"], render=True)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False
    episode_rewards = []
    memory, memory_mask, memory_indices = init_transformer_memory(config["transformer"], env.max_episode_steps, device)
    memory_length = config["transformer"]["memory_length"]
    t = 0
    obs = env.reset()
    G = env.create_networkx_graph()
    NUM_NODES = env.num_nodes
    Side = int(NUM_NODES**0.5)
    pos = {i: (i % Side, Side - 1 - i // Side) for i in range(NUM_NODES)}
    env.plot_graph(G,  pos, "test2.png")
    start_time = time.time()
    while not done:
        # Prepare observation and memory
        obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
        in_memory = memory[0, memory_indices[t].unsqueeze(0)]
        t_ = max(0, min(t, memory_length - 1))
        mask = memory_mask[t_].unsqueeze(0)
        indices = memory_indices[t].unsqueeze(0)
        # Render environment
        env.render()
        # Forward model
        policy, value, new_memory = model(obs, in_memory, mask, indices) #in_memory ([1, 64, 3, 512]) #indices 64
        # print(in_memory[0][0][0])#
        memory[:, t] = new_memory
        # Sample action
        action = []
        for action_branch in policy:
            action.append(action_branch.probs.argmax().item())
        # Step environemnt
        # print("next step is to go to ",int(action[0]))
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        t += 1
    
    # after done, render last state
    env.render()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"函數執行完成，結果是 ")
    print(f"函數執行耗時：{elapsed_time} 秒")   
    euler_path = env.path
    euler_path = [(str(euler_path[i]), str(euler_path[i + 1])) for i in range(len(euler_path) - 1)]
    repeat_path = euler_path
    if env.ok == 0:
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S") 
        with open('verify.txt', 'a') as file:
                file.write(f'\n{formatted_now}:{env.path}') 
        generate_gif_from_path(env.create_networkx_graph() , env.get_pos(), euler_path, repeat_path)

    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))
    print("Result is ",env.ok)
    env.close()
    return env.ok, info

if __name__ == "__main__":
    model_path = load_model_path() 
    total_reward = 0
    total_length = 0
    Test_times = 100
    for i in range(Test_times):
        test, info = main(model_path,i)
        print("return test",test)
        total_reward += info["reward"]
        total_length += info["length"]

    average_reward = total_reward / Test_times  # 假设循环次数是5
    average_length = total_length / Test_times  # 假设循环次数是5

    print("Average Reward:", average_reward)
    print("Average Length:", average_length)

# %%
