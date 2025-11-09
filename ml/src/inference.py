# File: ml/src/inference.py

"""
Main Inference Pipeline: Orchestrates all ML models
Real-time processing of webcam, typing, and cursor data
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional
import threading
import queue
import torch
import os

from emotion_cnn import EmotionCNN, FaceDetector
from focus_rnn import FocusTracker
from rag_retriever import InterventionRetriever, generate_intervention_text
from drl_agent import InterventionAgent


class NeuroFlowPipeline:
    """
    Main ML pipeline for NeuroFlow
    Processes all inputs and coordinates ML models
    """
    def __init__(self, model_dir: Optional[str] = None, kb_path: Optional[str] = None):
        print("Initializing NeuroFlow ML Pipeline...")
        # Resolve default paths relative to the ml/ folder
        ml_root = os.path.dirname(os.path.dirname(__file__))
        if not model_dir:
            model_dir = os.path.join(ml_root, 'models')
        if not kb_path:
            kb_path = os.path.join(ml_root, 'data', 'interventions.json')
        model_dir = os.path.abspath(model_dir)
        kb_path = os.path.abspath(kb_path)
        
        # Load models
        self.emotion_model = EmotionCNN()
        try:
            self.emotion_model.load_state_dict(
                torch.load(os.path.join(model_dir, 'emotion_cnn.pth'), map_location='cpu')
            )
            print("Loaded Emotion CNN weights")
        except Exception as e:
            print(f"Using untrained Emotion CNN (no weights at {os.path.join(model_dir, 'emotion_cnn.pth')})")
        
        self.face_detector = FaceDetector()
        
        self.focus_tracker = FocusTracker(
            sequence_length=30,
            model_path=os.path.join(model_dir, 'focus_rnn.pth')
        )
        print("Initialized Focus RNN tracker")
        
        self.intervention_retriever = InterventionRetriever(knowledge_base_path=kb_path)
        print("Loaded RAG System")
        
        self.drl_agent = InterventionAgent(
            model_path=os.path.join(model_dir, 'drl_agent.pth')
        )
        print("Loaded DRL Agent")
        
        # State tracking
        self.current_state = {
            'stress': 0.0,
            'focus': 0.5,
            'fatigue': 0.0,
            'emotions': {},
            'attention_drift_risk': 0.0
        }
        
        self.last_intervention_time = time.time()
        self.session_start_time = time.time()
        
        # Performance metrics
        self.fps = 0
        self.processing_times = []
        
        print("NeuroFlow Pipeline Ready!")
    
    def process_frame(self, frame: np.ndarray, 
                     keystroke: bool = False,
                     cursor_pos: Optional[tuple] = None) -> Dict:
        """
        Process single frame from webcam
        
        Args:
            frame: BGR image from cv2
            keystroke: whether a key was pressed this frame
            cursor_pos: (x, y) cursor position
        
        Returns:
            dict with all ML outputs for frontend
        """
        start_time = time.time()
        
        # 1. Face detection and emotion recognition (CNN)
        face_crop, blink_detected, head_pose = self.face_detector.detect_face(frame)
        
        emotion_result = None
        if face_crop is not None:
            emotion_result = self.emotion_model.predict_emotion(face_crop)
            
            # Update current state
            self.current_state['stress'] = emotion_result['stress']
            self.current_state['focus'] = emotion_result['focus']
            self.current_state['emotions'] = emotion_result['emotions']
        
        # 2. Update focus tracker (RNN)
        self.focus_tracker.update(
            emotion_data=emotion_result,
            blink=blink_detected,
            keystroke=keystroke,
            cursor_pos=cursor_pos
        )
        
        # 3. Predict focus and attention drift (RNN)
        focus_prediction = self.focus_tracker.predict_focus()
        self.current_state['attention_drift_risk'] = focus_prediction['attention_drift_risk']
        
        # 4. Decide if intervention is needed (DRL)
        time_since_last_intervention = (time.time() - self.last_intervention_time) / 60.0
        
        state_vector = self.drl_agent.get_state_vector(
            self.current_state,
            time_since_last_intervention
        )
        
        action = self.drl_agent.select_action(state_vector, training=False)
        intervention_params = self.drl_agent.get_intervention_intensity(action)
        
        # 5. Retrieve intervention if needed (RAG)
        intervention = None
        if intervention_params['should_intervene']:
            interventions = self.intervention_retriever.retrieve_interventions(
                self.current_state,
                top_k=1
            )
            if interventions:
                intervention = interventions[0]
                intervention['text'] = generate_intervention_text(intervention)
                intervention['intensity'] = intervention_params['intensity']
                self.last_intervention_time = time.time()
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # Construct output
        output = {
            # Core metrics
            'stress': float(self.current_state['stress']),
            'focus': float(focus_prediction['focus_probability']),
            'fatigue': float(self.current_state.get('fatigue', 0)),
            
            # Detailed emotion breakdown
            'emotions': self.current_state['emotions'] if emotion_result else {},
            'dominant_emotion': emotion_result['dominant_emotion'] if emotion_result else 'unknown',
            
            # Focus details
            'attention_drift_risk': float(focus_prediction['attention_drift_risk']),
            'focus_trend': focus_prediction['focus_trend'],
            'focus_confidence': float(focus_prediction['confidence']),
            
            # Physiological metrics
            'blink_rate': float(focus_prediction['blink_rate']),
            'blink_detected': bool(blink_detected),
            
            # Typing metrics
            'typing_consistency': float(focus_prediction['typing_consistency']),
            'keystroke_detected': bool(keystroke),
            
            # Head pose
            'head_pose': head_pose if head_pose else None,
            'face_detected': face_crop is not None,
            
            # Intervention
            'intervention': intervention,
            'intervention_action': int(action),
            
            # Session info
            'session_duration_minutes': (time.time() - self.session_start_time) / 60.0,
            'time_since_last_intervention_minutes': time_since_last_intervention,
            
            # Performance
            'processing_time_ms': float(processing_time),
            'avg_processing_time_ms': float(np.mean(self.processing_times)),
            
            # Timestamp
            'timestamp': time.time()
        }
        
        return output
    
    def get_session_summary(self) -> Dict:
        """Get summary statistics for current session"""
        return {
            'session_duration_minutes': (time.time() - self.session_start_time) / 60.0,
            'avg_stress': np.mean([h.get('stress_before', 0) 
                                  for h in self.drl_agent.intervention_history]) 
                         if self.drl_agent.intervention_history else 0,
            'avg_focus': np.mean([h.get('focus_before', 0.5) 
                                for h in self.drl_agent.intervention_history])
                        if self.drl_agent.intervention_history else 0.5,
            'total_interventions': len(self.drl_agent.intervention_history),
            'intervention_success_rate': np.mean(self.drl_agent.success_history)
                                        if self.drl_agent.success_history else 0,
        }
    
    def record_intervention_outcome(self, user_response: str, 
                                   state_before: Dict, 
                                   state_after: Dict,
                                   action: int):
        """
        Record user's response to intervention for learning
        
        Args:
            user_response: 'followed', 'accepted', 'dismissed', 'ignored'
            state_before: mental state before intervention
            state_after: mental state after intervention
            action: intervention action taken
        """
        # Calculate reward
        reward = self.drl_agent.calculate_reward(
            state_before, state_after, action, user_response
        )
        
        # Store for learning
        state_vector_before = self.drl_agent.get_state_vector(
            state_before,
            (self.last_intervention_time - self.session_start_time) / 60.0
        )
        
        state_vector_after = self.drl_agent.get_state_vector(
            state_after,
            (time.time() - self.last_intervention_time) / 60.0
        )
        
        self.drl_agent.remember(
            state_vector_before,
            action,
            reward,
            state_vector_after,
            done=False
        )
        
        # Store in history
        self.drl_agent.intervention_history.append({
            'stress_before': state_before['stress'],
            'focus_before': state_before['focus'],
            'stress_after': state_after['stress'],
            'focus_after': state_after['focus'],
            'response': user_response,
            'reward': reward
        })
        
        success = user_response in ['followed', 'accepted']
        self.drl_agent.success_history.append(1.0 if success else 0.0)


class WebcamCapture:
    """Thread-safe webcam capture"""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Continuous capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame(self):
        """Get latest frame"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop capture"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()


if __name__ == '__main__':
    # Test pipeline with webcam
    print("Starting NeuroFlow Test...")
    
    pipeline = NeuroFlowPipeline()
    webcam = WebcamCapture(camera_id=0)
    webcam.start()
    
    print("\nPress 'q' to quit, 'i' to simulate intervention response")
    
    last_process_time = time.time()
    
    try:
        while True:
            # Get frame
            frame = webcam.get_frame()
            if frame is None:
                continue
            
            # Process at 1 Hz
            current_time = time.time()
            if current_time - last_process_time < 1.0:
                continue
            
            last_process_time = current_time
            
            # Simulate keystroke and cursor
            keystroke = np.random.random() > 0.7
            cursor_pos = (np.random.randint(0, 1920), np.random.randint(0, 1080))
            
            # Process frame
            result = pipeline.process_frame(frame, keystroke, cursor_pos)
            
            # Display results
            print(f"\n{'='*60}")
            print(f"Stress: {result['stress']:.2f} | Focus: {result['focus']:.2f} | "
                  f"Drift Risk: {result['attention_drift_risk']:.2f}")
            print(f"Emotion: {result['dominant_emotion']} | "
                  f"Blink Rate: {result['blink_rate']:.1f} bpm")
            
            if result['intervention']:
                print(f"\n🔔 INTERVENTION SUGGESTED:")
                print(f"   {result['intervention']['name']} ({result['intervention']['intensity']})")
                print(f"   {result['intervention']['description']}")
            
            print(f"Processing: {result['processing_time_ms']:.1f}ms")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        webcam.stop()
        print("\nStopped!")
        
        # Print session summary
        summary = pipeline.get_session_summary()
        print(f"\nSession Summary:")
        print(f"  Duration: {summary['session_duration_minutes']:.1f} minutes")
        print(f"  Avg Stress: {summary['avg_stress']:.2f}")
        print(f"  Avg Focus: {summary['avg_focus']:.2f}")
        print(f"  Total Interventions: {summary['total_interventions']}")