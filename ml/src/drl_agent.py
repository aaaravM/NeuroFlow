# File: ml/src/drl_agent.py

"""
DRL Agent: Optimizes when and how to deliver interventions
Uses Deep Q-Network (DQN) to learn user preferences
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNetwork(nn.Module):
    """Deep Q-Network for intervention timing"""
    def __init__(self, state_size=15, action_size=4):
        super(DQNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.fc(x)


class InterventionAgent:
    """
    DRL Agent that learns optimal intervention timing
    
    State: [stress, focus, fatigue, time_since_last_intervention, 
            user_reaction_history, intervention_success_rate, ...]
    
    Actions:
        0: No intervention (wait)
        1: Gentle nudge (low intensity)
        2: Standard intervention (medium intensity)
        3: Strong intervention (high intensity)
    """
    def __init__(self, state_size=15, action_size=4, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Q-Networks
        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Load model if available
        if model_path:
            try:
                self.q_network.load_state_dict(torch.load(model_path))
                self.target_network.load_state_dict(self.q_network.state_dict())
                print(f"Loaded DRL agent from {model_path}")
            except:
                print("Could not load model, using random initialization")
        
        # Tracking
        self.last_intervention_time = 0
        self.intervention_history = deque(maxlen=10)
        self.success_history = deque(maxlen=10)
        
    def get_state_vector(self, mental_state, time_elapsed):
        """
        Convert current context to state vector
        
        Args:
            mental_state: dict with 'stress', 'focus', 'fatigue', etc.
            time_elapsed: minutes since last intervention
        """
        state = [
            mental_state.get('stress', 0),
            mental_state.get('focus', 0.5),
            mental_state.get('fatigue', 0),
            mental_state.get('attention_drift_risk', 0),
            time_elapsed / 60.0,  # Normalize to hours
            
            # Historical features
            len(self.intervention_history) / 10.0,
            np.mean(self.success_history) if self.success_history else 0.5,
            
            # Time-of-day features (if available)
            mental_state.get('hour_of_day', 12) / 24.0,
            mental_state.get('day_of_week', 3) / 7.0,
            
            # Trend features
            self._get_stress_trend(),
            self._get_focus_trend(),
            
            # Physical features
            mental_state.get('blink_rate', 17) / 30.0,
            mental_state.get('typing_consistency', 0.5),
            mental_state.get('cursor_activity', 0.5),
            
            # Productivity proxy
            mental_state.get('productive_time_ratio', 0.5)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _get_stress_trend(self):
        """Calculate stress trend from recent interventions"""
        if len(self.intervention_history) < 2:
            return 0.0
        
        recent = [h['stress_before'] for h in list(self.intervention_history)[-3:]]
        if len(recent) < 2:
            return 0.0
        
        trend = (recent[-1] - recent[0]) / len(recent)
        return np.clip(trend, -1, 1)
    
    def _get_focus_trend(self):
        """Calculate focus trend from recent interventions"""
        if len(self.intervention_history) < 2:
            return 0.0
        
        recent = [h['focus_before'] for h in list(self.intervention_history)[-3:]]
        if len(recent) < 2:
            return 0.0
        
        trend = (recent[-1] - recent[0]) / len(recent)
        return np.clip(trend, -1, 1)
    
    def select_action(self, state_vector, training=False):
        """
        Select action using epsilon-greedy policy
        
        Returns:
            action: 0=wait, 1=gentle, 2=standard, 3=strong
        """
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, state_before, state_after, action, user_response):
        """
        Calculate reward for intervention
        
        Args:
            state_before: mental state before intervention
            state_after: mental state after intervention
            action: 0=wait, 1=gentle, 2=standard, 3=strong
            user_response: 'accepted', 'dismissed', 'ignored', 'followed'
        """
        reward = 0.0
        
        # Reward for improvement in mental state
        stress_improvement = state_before.get('stress', 0) - state_after.get('stress', 0)
        focus_improvement = state_after.get('focus', 0) - state_before.get('focus', 0)
        
        reward += stress_improvement * 2.0  # Stress reduction is valuable
        reward += focus_improvement * 3.0   # Focus improvement is very valuable
        
        # User response rewards
        response_rewards = {
            'followed': 1.0,      # User completed intervention
            'accepted': 0.5,      # User acknowledged but may not have completed
            'dismissed': -0.3,    # User explicitly dismissed
            'ignored': -0.5       # User ignored completely
        }
        reward += response_rewards.get(user_response, 0)
        
        # Penalty for interrupting unnecessarily
        if action > 0:  # Any intervention
            # Small penalty for interrupting
            reward -= 0.1
            
            # Additional penalty if state was already good
            if state_before.get('focus', 0) > 0.7 and state_before.get('stress', 0) < 0.3:
                reward -= 0.3
        
        # Penalty for being too aggressive
        if action == 3 and state_before.get('stress', 0) < 0.5:
            reward -= 0.2
        
        # Reward for well-timed waiting
        if action == 0:
            if state_before.get('focus', 0) > 0.7:
                reward += 0.2  # Good decision to not interrupt
        
        return np.clip(reward, -2.0, 2.0)
    
    def get_intervention_intensity(self, action):
        """
        Convert action to intervention parameters
        
        Returns:
            dict with 'should_intervene', 'intensity', 'duration'
        """
        intensities = {
            0: {'should_intervene': False, 'intensity': 'none', 'duration': 0},
            1: {'should_intervene': True, 'intensity': 'gentle', 'duration': 15},
            2: {'should_intervene': True, 'intensity': 'standard', 'duration': 30},
            3: {'should_intervene': True, 'intensity': 'strong', 'duration': 60}
        }
        return intensities[action]
    
    def save_model(self, path='ml/models/drl_agent.pth'):
        """Save Q-network"""
        torch.save(self.q_network.state_dict(), path)
        print(f"DRL agent saved to {path}")


def train_drl_agent(agent, num_episodes=1000):
    """
    Train DRL agent in simulated environment
    For hackathon: use this with synthetic data, then fine-tune with real users
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Simulate user session
        episode_reward = 0
        time_elapsed = 0
        
        # Initialize mental state
        mental_state = {
            'stress': np.random.random(),
            'focus': np.random.random(),
            'fatigue': np.random.random(),
            'hour_of_day': np.random.randint(0, 24),
            'attention_drift_risk': np.random.random()
        }
        
        for step in range(60):  # 60-minute session
            # Get state vector
            state_vector = agent.get_state_vector(mental_state, time_elapsed)
            
            # Select action
            action = agent.select_action(state_vector, training=True)
            
            # Simulate intervention and response
            intensity = agent.get_intervention_intensity(action)
            
            # Simulate state changes and user response
            next_state = mental_state.copy()
            if intensity['should_intervene']:
                # Simulate intervention effect
                next_state['stress'] = max(0, mental_state['stress'] - np.random.uniform(0.1, 0.3))
                next_state['focus'] = min(1, mental_state['focus'] + np.random.uniform(0.1, 0.3))
                
                # Simulate user response
                user_response = np.random.choice(
                    ['followed', 'accepted', 'dismissed', 'ignored'],
                    p=[0.4, 0.3, 0.2, 0.1]
                )
            else:
                # Natural drift
                next_state['stress'] = min(1, mental_state['stress'] + np.random.uniform(-0.05, 0.1))
                next_state['focus'] = max(0, mental_state['focus'] - np.random.uniform(0, 0.1))
                user_response = 'none'
            
            # Calculate reward
            reward = agent.calculate_reward(mental_state, next_state, action, user_response)
            episode_reward += reward
            
            # Store experience
            next_state_vector = agent.get_state_vector(next_state, time_elapsed + 1)
            done = (step == 59)
            agent.remember(state_vector, action, reward, next_state_vector, done)
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            # Update state
            mental_state = next_state
            time_elapsed += 1
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent


if __name__ == '__main__':
    print("Training DRL Agent...")
    agent = InterventionAgent()
    
    # Train agent
    trained_agent = train_drl_agent(agent, num_episodes=500)
    
    # Save model
    trained_agent.save_model()
    
    print("\nTesting trained agent...")
    # Test with high stress scenario
    test_state = {
        'stress': 0.8,
        'focus': 0.3,
        'fatigue': 0.6,
        'hour_of_day': 14,
        'attention_drift_risk': 0.7
    }
    
    state_vector = trained_agent.get_state_vector(test_state, time_elapsed=30)
    action = trained_agent.select_action(state_vector, training=False)
    intensity = trained_agent.get_intervention_intensity(action)
    
    print(f"\nTest State: {test_state}")
    print(f"Agent Decision: {intensity}")
    print(f"Action: {action} (0=wait, 1=gentle, 2=standard, 3=strong)")