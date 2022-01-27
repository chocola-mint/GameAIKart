import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNet(nn.Module):
    # TODO(Lab-4): Q-Network architecture.
    def __init__(self, input_shape, n_actions):
        super(QNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.fc(conv_out)
        return out

class DeepQNetwork():
    def __init__(
        self,
        n_actions,
        input_shape,
        qnet,
        device,
        learning_rate = 2e-4,
        reward_decay = 0.99,
        replace_target_iter = 1000,
        memory_size = 10000,
        batch_size = 32,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.learn_step_counter = 0
        self.init_memory()

        # Network
        self.qnet_eval = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target.eval()
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon=0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions_value = self.qnet_eval.forward(state)
        #print(actions_value)
        if np.random.uniform() > epsilon:   # greedy
            action = actions_value[0].data.cpu().numpy()[0]
        else:   # random
            action = np.random.uniform(-1.0, 1.0)
        return np.array([[action]], dtype="float32")

    def learn(self):
        # TODO(Lab-5): DQN core algorithm.
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())
        
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        b_s = torch.FloatTensor(self.memory["s"][sample_index]).to(self.device)
        b_a = torch.LongTensor(self.memory["a"][sample_index]).to(self.device)
        b_a = torch.zeros_like(b_a)
        b_r = torch.FloatTensor(self.memory["r"][sample_index]).to(self.device)
        b_s_ = torch.FloatTensor(self.memory["s_"][sample_index]).to(self.device)
        b_d = torch.FloatTensor(self.memory["done"][sample_index]).to(self.device)
        print(f"b_a = {b_a}")
        q_curr_eval = self.qnet_eval(b_s).gather(1, b_a)
        q_next_target = self.qnet_target(b_s_).detach()
        #next_state_values = q_next_target.max(1)[0].view(-1, 1) # DQN
        q_next_eval = self.qnet_eval(b_s_).detach()
        next_state_values = q_next_target.gather(1, q_next_eval.max(1)[1].unsqueeze(1)) # DDQN
        q_curr_recur = b_r + (1-b_d) * self.gamma * next_state_values

        self.loss = F.smooth_l1_loss(q_curr_eval, q_curr_recur)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        return self.loss.detach().cpu().numpy()

    def init_memory(self):
        self.memory = {
            "s": np.zeros((self.memory_size, *self.input_shape)),
            "a": np.zeros((self.memory_size, 1)),
            "r": np.zeros((self.memory_size, 1)),
            "s_": np.zeros((self.memory_size, *self.input_shape)),
            "done": np.zeros((self.memory_size, 1)),
        }

    def store_transition(self, s, a, r, s_, d):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if self.memory_counter <= self.memory_size:
            index = self.memory_counter % self.memory_size
        else:
            index = np.random.randint(self.memory_size)
        self.memory["s"][index] = s
        self.memory["a"][index] = np.array(a).reshape(-1,1)
        self.memory["r"][index] = np.array(r).reshape(-1,1)
        self.memory["s_"][index] = s_
        self.memory["done"][index] = np.array(d).reshape(-1,1)
        self.memory_counter += 1
    
    def save_load_model(self, op, path="save", fname="qnet.pt"):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, fname)
        if op == "save":
            torch.save(self.qnet_eval.state_dict(), file_path)
        elif op == "load":
            self.qnet_eval.load_state_dict(torch.load(file_path, map_location=self.device))
            self.qnet_target.load_state_dict(torch.load(file_path, map_location=self.device))

class DQNAgent(DeepQNetwork):
    def __init__(self, action_space_n, stack_frames, img_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
        n_actions = action_space_n,
        input_shape = [stack_frames, *img_size],
        qnet = QNet,
        device = device,
        learning_rate = 2e-4, 
        reward_decay = 0.99,
        replace_target_iter = 1000, 
        memory_size = 10000,
        batch_size = 4,)
        print(self.qnet_eval)
        pass