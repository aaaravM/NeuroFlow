# File: ml/src/focus_rnn.py

"""
Focus RNN: Predicts attention drift using temporal sequences
Processes: emotion history, blink rate, typing rhythm, cursor movement
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time

class FocusRNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.3):
        super(FocusRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Focus probability [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            focus_prob: (batch_size, 1) - probability of being focused
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), 
            dim=1
        )
        
        # Weighted sum of LSTM outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)
        
        # Predict focus probability
        focus_prob = self.fc(context)
        
        return focus_prob, attention_weights


class FocusTracker:
    """
    Real-time focus tracking system
    Maintains temporal buffer and predicts attention drift
    """
    def __init__(self, sequence_length=30, model_path=None):
        self.sequence_length = sequence_length  # 30 seconds at 1Hz
        self.model = FocusRNN()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print(f"Loaded focus model from {model_path}")
            except:
                print("Could not load model, using random weights")
        
        # Temporal buffers
        self.emotion_buffer = deque(maxlen=sequence_length)
        self.blink_buffer = deque(maxlen=sequence_length)
        self.typing_buffer = deque(maxlen=sequence_length)
        self.cursor_buffer = deque(maxlen=sequence_length)
        
        # Baseline metrics
        self.baseline_blink_rate = 17  # blinks per minute
        self.last_update_time = time.time()
        
        # Typing tracking
        self.last_keystroke_time = 0
        self.keystroke_intervals = deque(maxlen=10)
    
    def update(self, emotion_data=None, blink=False, keystroke=False, cursor_pos=None):
        """
        Update buffers with new data point
        Call this every ~1 second
        """
        current_time = time.time()
        
        # Emotion features (7 dimensions)
        if emotion_data:
            emotion_vector = [
                emotion_data.get('stress', 0),
                emotion_data.get('focus', 0),
                emotion_data['emotions'].get('angry', 0),
                emotion_data['emotions'].get('fear', 0),
                emotion_data['emotions'].get('happy', 0),
                emotion_data['emotions'].get('neutral', 0),
                emotion_data['emotions'].get('sad', 0)
            ]
        else:
            emotion_vector = [0] * 7
        
        self.emotion_buffer.append(emotion_vector)
        
        # Blink tracking (1 dimension)
        self.blink_buffer.append(1 if blink else 0)
        
        # Typing rhythm (2 dimensions)
        if keystroke:
            if self.last_keystroke_time > 0:
                interval = current_time - self.last_keystroke_time
                self.keystroke_intervals.append(interval)
            self.last_keystroke_time = current_time
        
        typing_speed = len(self.keystroke_intervals) / 10.0  # normalized
        typing_variance = np.std(self.keystroke_intervals) if len(self.keystroke_intervals) > 1 else 0
        self.typing_buffer.append([typing_speed, typing_variance])
        
        # Cursor movement (2 dimensions)
        if cursor_pos:
            if len(self.cursor_buffer) > 0:
                last_pos = self.cursor_buffer[-1]
                movement = np.sqrt((cursor_pos[0] - last_pos[0])**2 + 
                                 (cursor_pos[1] - last_pos[1])**2)
            else:
                movement = 0
            self.cursor_buffer.append([cursor_pos[0]/1920, cursor_pos[1]/1080])  # normalized
        else:
            self.cursor_buffer.append([0, 0])
        
        self.last_update_time = current_time
    
    def predict_focus(self):
        """
        Predict current focus level and attention drift probability
        """
        if len(self.emotion_buffer) < 5:  # Need minimum data
            return {
                'focus_probability': 0.5,
                'attention_drift_risk': 0.0,
                'focus_trend': 'neutral',
                'confidence': 0.0
            }
        
        # Construct feature sequence
        sequence = []
        for i in range(min(self.sequence_length, len(self.emotion_buffer))):
            idx = -self.sequence_length + i if i < self.sequence_length else i
            
            features = (
                list(self.emotion_buffer[idx]) +  # 7 emotion features
                [self.blink_buffer[idx]] +         # 1 blink feature
                list(self.typing_buffer[idx]) +    # 2 typing features
                list(self.cursor_buffer[idx])      # 2 cursor features
            )
            sequence.append(features)
        
        # Pad sequence if too short
        while len(sequence) < self.sequence_length:
            sequence.insert(0, [0] * 12)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            focus_prob, attention_weights = self.model(input_tensor)
        
        focus_probability = float(focus_prob[0, 0])
        
        # Calculate attention drift risk (inverse of focus)
        attention_drift_risk = 1.0 - focus_probability
        
        # Determine trend
        if len(self.emotion_buffer) >= 10:
            recent_avg = np.mean([self.emotion_buffer[i][1] for i in range(-10, 0)])
            older_avg = np.mean([self.emotion_buffer[i][1] for i in range(-20, -10)])
            
            if recent_avg > older_avg + 0.1:
                trend = 'improving'
            elif recent_avg < older_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'neutral'
        
        # Calculate confidence based on data quality
        confidence = min(len(self.emotion_buffer) / self.sequence_length, 1.0)
        
        return {
            'focus_probability': focus_probability,
            'attention_drift_risk': attention_drift_risk,
            'focus_trend': trend,
            'confidence': confidence,
            'blink_rate': self._calculate_blink_rate(),
            'typing_consistency': self._calculate_typing_consistency()
        }
    
    def _calculate_blink_rate(self):
        """Calculate blinks per minute"""
        if len(self.blink_buffer) == 0:
            return 0
        
        blinks = sum(self.blink_buffer)
        duration_minutes = len(self.blink_buffer) / 60.0
        return blinks / duration_minutes if duration_minutes > 0 else 0
    
    def _calculate_typing_consistency(self):
        """Calculate typing rhythm consistency (0=erratic, 1=consistent)"""
        if len(self.keystroke_intervals) < 3:
            return 0.5
        
        variance = np.var(self.keystroke_intervals)
        consistency = np.exp(-variance)  # High variance = low consistency
        return float(consistency)


# Training function
def train_focus_rnn(train_loader, val_loader, num_epochs=50):
    """
    Train Focus RNN
    
    Expected data format:
    - X: (batch, sequence_length, 12) temporal features
    - y: (batch, 1) focus label (0=distracted, 1=focused)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FocusRNN().to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            predictions, _ = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            train_correct += (predicted_labels == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device).float()
                
                predictions, _ = model(sequences)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                predicted_labels = (predictions > 0.5).float()
                val_correct += (predicted_labels == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ml/models/focus_rnn.pth')
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
    
    return model


if __name__ == '__main__':
    # Test focus tracker
    tracker = FocusTracker(sequence_length=30)
    
    print("Simulating focus tracking...")
    for i in range(60):
        # Simulate data
        emotion_data = {
            'stress': np.random.random() * 0.3,
            'focus': 0.7 + np.random.random() * 0.3,
            'emotions': {
                'angry': 0.05, 'fear': 0.05, 'happy': 0.6,
                'neutral': 0.25, 'sad': 0.05
            }
        }
        
        tracker.update(
            emotion_data=emotion_data,
            blink=np.random.random() > 0.95,
            keystroke=np.random.random() > 0.7,
            cursor_pos=(np.random.randint(0, 1920), np.random.randint(0, 1080))
        )
        
        if i % 10 == 0:
            result = tracker.predict_focus()
            print(f"\nStep {i}:")
            print(f"  Focus Probability: {result['focus_probability']:.2f}")
            print(f"  Drift Risk: {result['attention_drift_risk']:.2f}")
            print(f"  Trend: {result['focus_trend']}")
            print(f"  Confidence: {result['confidence']:.2f}")
        
        time.sleep(0.1)