# File: ml/src/emotion_cnn.py

"""
Emotion CNN: Real-time facial emotion detection from webcam frames
Uses transfer learning from ResNet18 for fast inference
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from PIL import Image

class EmotionCNN(nn.Module):
    def __init__(self, num_emotions=7):
        super(EmotionCNN, self).__init__()
        # Use pretrained ResNet18 as backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace final layer for emotion classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_emotions)
        )
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def forward(self, x):
        return self.backbone(x)
    
    def predict_emotion(self, frame):
        """
        Predict emotion from a single frame
        Args:
            frame: numpy array (H, W, 3) in BGR format from cv2
        Returns:
            dict with emotion probabilities and stress/focus metrics
        """
        # Preprocess frame
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Inference
        self.eval()
        with torch.no_grad():
            outputs = self(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Convert to emotion dict
        emotion_scores = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, probabilities)
        }
        
        # Calculate stress and focus metrics
        stress_level = self._calculate_stress(emotion_scores)
        focus_level = self._calculate_focus(emotion_scores)
        
        return {
            'emotions': emotion_scores,
            'stress': stress_level,
            'focus': focus_level,
            'dominant_emotion': self.emotions[torch.argmax(probabilities).item()]
        }
    
    def _calculate_stress(self, emotion_scores):
        """Calculate stress level from emotion distribution"""
        # High stress: angry, fear, disgust
        stress = (
            emotion_scores['angry'] * 1.0 +
            emotion_scores['fear'] * 0.9 +
            emotion_scores['disgust'] * 0.7 +
            emotion_scores['sad'] * 0.5
        )
        return min(stress, 1.0)
    
    def _calculate_focus(self, emotion_scores):
        """Calculate focus level from emotion distribution"""
        # High focus: neutral, slight happiness
        focus = (
            emotion_scores['neutral'] * 1.0 +
            emotion_scores['happy'] * 0.6 -
            emotion_scores['surprise'] * 0.3 -
            emotion_scores['fear'] * 0.5
        )
        return max(0.0, min(focus, 1.0))


class FaceDetector:
    """Detect and extract face from frame"""
    def __init__(self):
        # Use OpenCV's Haar Cascade for fast face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_face(self, frame):
        """
        Detect face and eyes in frame
        Returns: (face_crop, blink_detected, head_pose)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, False, None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[y:y+h, x:x+w]
        
        # Detect eyes for blink detection
        face_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5)
        blink_detected = len(eyes) < 2  # Blink if less than 2 eyes visible
        
        # Simple head pose (could be improved with dlib)
        head_pose = {
            'x_center': x + w/2,
            'y_center': y + h/2,
            'width': w,
            'height': h
        }
        
        return face_crop, blink_detected, head_pose


# Training function (for your notebooks)
def train_emotion_cnn(train_loader, val_loader, num_epochs=20):
    """
    Train the emotion CNN
    Use FER2013 or AffectNet dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ml/models/emotion_cnn.pth')
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
    
    return model


if __name__ == '__main__':
    # Quick test with webcam
    model = EmotionCNN()
    # Load pretrained weights if available
    try:
        model.load_state_dict(torch.load('ml/models/emotion_cnn.pth'))
        print("Loaded pretrained model")
    except:
        print("No pretrained model found, using random weights")
    
    face_detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        face_crop, blink, head_pose = face_detector.detect_face(frame)
        
        if face_crop is not None:
            result = model.predict_emotion(face_crop)
            
            # Display results on frame
            cv2.putText(frame, f"Emotion: {result['dominant_emotion']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Stress: {result['stress']:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Focus: {result['focus']:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('NeuroFlow Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()