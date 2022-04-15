import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self,file_path, file_name='dqn_model.pth'):
        model_folder_path = file_path
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state =  torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values using current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] +self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Qnew = r + y * max(predicted Q value using next state) -> if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

class Dueling_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Dueling_DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1_adv = nn.Linear(hidden_size, hidden_size)
        self.linear1_val = nn.Linear(hidden_size, hidden_size)

        self.linear2_adv = nn.Linear(hidden_size, output_size)
        self.linear2_val = nn.Linear(hidden_size, 1)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        value = F.relu(self.linear1_adv(y))
        adv = F.relu(self.linear1_val(y))

        value = self.linear2_adv(value)
        adv = self.linear2_adv(adv)

        return value + (adv - adv.mean())

    def save(self,file_path, file_name='dueling_model.pth'):
        model_folder_path = file_path
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
