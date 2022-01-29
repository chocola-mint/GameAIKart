import logging
from tkinter import Image # you can use functions in logging: debug, info, warning, error, critical, log
import config
import PAIA
from demo import Demo
from ppo import PolicyNet, ValueNet, PPO
import torch
import numpy as np
from torchvision import transforms
import os
from config import MAX_EPISODES
SAVE_DIR = "agents/ppo"
# PPO_MODE = "train"
PPO_MODE = "play"

class MLPlay:
    def __init__(self):
        self.demo = Demo.create_demo() # create a replay buffer
        self.step_number = 0 # Count the step, not necessarily
        self.episode_number = 1 # Count the episode, not necessarily

        self.max_step = 300
        self.progress = 0
        self.s_dim = 64 * 64 * 3
        self.a_dim = 3 # Number of actions (use argmax on the actions array to get the most confident action)
        self.gamma=0.99
        self.lamb=0.95
        self.policy_net = PolicyNet(self.s_dim, self.a_dim)
        self.value_net = ValueNet(self.s_dim)
        self.agent = PPO(self.policy_net, self.value_net)
        self.device = "cpu"
        self.mean_total_reward = 0
        self.mean_length = 0
        self.total_reward = 0
        self.length = 0
        
        self.mb_states = np.zeros((self.max_step, self.s_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.max_step, self.a_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if PPO_MODE != "train":
            model_path = os.path.join(SAVE_DIR, "model.pt")
            if os.path.exists(model_path):
                print("Loading the model ... ", end="")
                checkpoint = torch.load(model_path)
                self.policy_net.load_state_dict(checkpoint["PolicyNet"])
                print("Done.")
            else:
                print('ERROR: No model saved')

    #Compute discounted return
    def compute_discounted_return(self, rewards, last_value):
        #TODO(Lab-3): Compute discounted return
        returns = np.zeros_like(rewards)
        n_step = len(rewards)

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                returns[t] = rewards[t] + self.gamma * last_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]
        return returns
    
    #Compute generalized advantage estimation (Optional)
    def compute_gae(self, rewards, values, last_value):
        advs = np.zeros_like(rewards)
        n_step = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                next_value = last_value
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma*next_value - values[t]
            advs[t] = last_gae_lam = delta + self.gamma*self.lamb*last_gae_lam

        return advs + values
    def decision(self, state: PAIA.State) -> PAIA.Action:
        #print("client: decision")
        '''
        Implement yor main algorithm here.
        Given a state input and make a decision to output an action
        '''
        # Implement Your Algorithm
        # Note: You can use PAIA.image_to_array() to convert
        #       state.observation.images.front.data and 
        #       state.observation.images.back.data to numpy array (range from 0 to 1)
        #       For example: img_array = PAIA.image_to_array(state.observation.images.front.data)
        with torch.no_grad():
            img_array = PAIA.image_to_array(state.observation.images.front.data)
            image = torch.FloatTensor(img_array)
            image = image.swapaxes(0, 2)
            image = transforms.Compose([transforms.Resize((64, 64))])(image)
            image = image.flatten(0)

        self.step_number += 1
        #logging.info('Episode: ' + str(self.episode_number) + ', Step: ' + str(self.step_number))

        img_suffix = str(self.episode_number) + '_' + str(self.step_number)
        logging.debug(PAIA.state_info(state, img_suffix))
        if self.step_number >= self.max_step:
            state.event = PAIA.Event.EVENT_TIMEOUT

        if state.event == PAIA.Event.EVENT_NONE:
            # Continue the game
            # You can decide your own action (change the following action to yours)
            reward = 0
            if state.observation.progress > self.progress:
                reward = (state.observation.progress - self.progress)*1000
                self.progress = state.observation.progress
            rL = (state.observation.rays.L.distance if state.observation.rays.L.hit else 1.0) - 0.1
            rR = (state.observation.rays.R.distance if state.observation.rays.R.hit else 1.0) - 0.1
            rF = (state.observation.rays.F.distance if state.observation.rays.F.hit else 1.0) - 0.1
            if rL < 0: rL -= 2.0
            if rR < 0: rR -= 2.0
            if rF < 0 : rF -= 2.0

            rV = state.observation.velocity / 20.0 - 0.2
            reward += rL + rR + rF + rV
            if reward < -1: reward = -1
            
            #reward -= 1
            with torch.no_grad():

                if PPO_MODE == "train":
                    
                    step = self.step_number
                    episode_len = self.max_step
                    if step > 0:
                        self.mb_rewards[step - 1] = reward
                    state_tensor = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32, device=self.device)
                    action, a_logp = self.policy_net(state_tensor)
                    value = self.value_net(state_tensor)

                    action = action.cpu().numpy()[0]
                    a_logp = a_logp.cpu().numpy()
                    value = value.cpu().numpy()

                    self.mb_states[step] = image
                    self.mb_actions[step] = action
                    self.mb_a_logps[step] = a_logp
                    self.mb_values[step] = value
                else:
                    state_tensor = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32, device='cpu')
                    action = self.policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
                    self.total_reward += reward
                    self.length += 1

            #print(f"Reward = {reward}")
            #print(f"Action = {action}")
            action_id = torch.argmax(torch.nn.Softmax(dim=0)(torch.FloatTensor(action)))
            #print(f"Selected {action_id}", flush=True)
            if action_id == 0 :
                action = PAIA.create_action_object(acceleration=True, brake=False, steering=0.0)
            elif action_id == 1 :
                action = PAIA.create_action_object(acceleration=True, brake=False, steering=1.0)
            elif action_id == 2:
                action = PAIA.create_action_object(acceleration=True, brake=False, steering=-1.0)
            elif action_id == 3:
                action = PAIA.create_action_object(acceleration=False, brake=True, steering=0.0)
            elif action_id == 4:
                action = PAIA.create_action_object(acceleration=False, brake=True, steering=1.0)
            else:
                action = PAIA.create_action_object(acceleration=False, brake=True, steering=-1.0)

            # You can save the step to the replay buffer (self.demo)
            self.demo.create_step(state=state, action=action)
        elif state.event == PAIA.Event.EVENT_RESTART:
            # You can do something when the game restarts by someone
            # You can decide your own action (change the following action to yours)
            action = PAIA.create_action_object(acceleration=True, brake=False, steering=0.0)
            # You can start a new episode and save the step to the replay buffer (self.demo)
            self.demo.create_episode()
            self.demo.create_step(state=state, action=action)
        elif state.event != PAIA.Event.EVENT_NONE:
            # You can do something when the game (episode) ends
            want_to_restart = True # Uncomment if you want to restart
            # want_to_restart = False # Uncomment if you want to finish
            if self.episode_number < config.MAX_EPISODES and want_to_restart:
                # Do something when restart
                if PPO_MODE == "train":
                    #Compute returns
                    with torch.no_grad():
                        last_value = self.value_net(
                            torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32, device=self.device)
                        ).cpu().numpy()
                        episode_len = self.max_step
                        mb_returns = self.compute_discounted_return(self.mb_rewards[:episode_len], last_value)
                        mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = self.mb_states[:episode_len], \
                        self.mb_actions[:episode_len], \
                        self.mb_a_logps[:episode_len], \
                        self.mb_values[:episode_len], \
                        mb_returns, \
                        self.mb_rewards[:episode_len]
                        mb_advs = mb_returns - mb_values
                        mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)
                    
                    #Train the model using the collected data
                    pg_loss, v_loss, ent = self.agent.train(mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps)
                    self.mean_total_reward += mb_rewards.sum()
                    self.mean_length += len(mb_states)
                    print(f"[Episode {self.episode_number:4d}] total reward = {mb_rewards.sum():.6f}, length = {len(mb_states):d}")
                    #Show the current result & save the model
                    if self.episode_number % 100 == 0:
                        print("\n[{:5d} / {:5d}]".format(self.episode_number, MAX_EPISODES))
                        print("----------------------------------")
                        print("actor loss = {:.6f}".format(pg_loss))
                        print("critic loss = {:.6f}".format(v_loss))
                        print("entropy = {:.6f}".format(ent))
                        print("mean return = {:.6f}".format(self.mean_total_reward / 200))
                        print("mean length = {:.2f}".format(self.mean_length / 200))
                        print("\nSaving the model ... ", end="")
                        torch.save({
                            "it": self.episode_number,
                            "PolicyNet": self.policy_net.state_dict(),
                            "ValueNet": self.value_net.state_dict()
                        }, os.path.join(SAVE_DIR, "model.pt"))
                        print("Done.")
                        print()
                        self.mean_total_reward = 0
                        self.mean_length = 0
                else:
                    print(f"[Evaluation] Total reward = {self.total_reward:.6f}, length = {self.length:d}", flush=True)

                action = PAIA.create_action_object(command=PAIA.Command.COMMAND_RESTART)
                self.episode_number += 1
                self.step_number = 0
                self.progress = 0
                # You can save the step to the replay buffer (self.demo)
                self.demo.create_step(state=state, action=action)
            else:
                # Do something when finish
                action = PAIA.create_action_object(command=PAIA.Command.COMMAND_FINISH)
                # You can save the step to the replay buffer (self.demo)
                self.demo.create_step(state=state, action=action)
                # You can export your replay buffer
                self.demo.export('kart.paia')
        
        logging.debug(PAIA.action_info(action))

        return action