"""
RAG Retriever: Retrieves cognitive interventions from knowledge base
Uses FAISS for fast semantic search
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

class InterventionRetriever:
    def __init__(self, knowledge_base_path='ml/data/interventions.json'):
        self.knowledge_base_path = knowledge_base_path
        
        # Load sentence embedding model
        print("Loading SentenceTransformer model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384-dim embeddings
        
        # Load or create knowledge base
        self.interventions = self._load_knowledge_base()
        
        # Create FAISS index
        self.index = None
        self._build_index()
    
    def _load_knowledge_base(self):
        """Load intervention knowledge base"""
        if os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, 'r') as f:
                return json.load(f)
        else:
            # Create default knowledge base
            interventions = self._create_default_knowledge_base()
            os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(interventions, f, indent=2)
            return interventions
    
    def _create_default_knowledge_base(self):
        """Create comprehensive intervention database"""
        return [
            {
                "id": "breathing_box",
                "name": "Box Breathing",
                "description": "4-4-4-4 breathing pattern to reduce stress",
                "duration_seconds": 60,
                "effectiveness_stress": 0.9,
                "effectiveness_focus": 0.6,
                "category": "breathing",
                "instructions": "Inhale for 4 seconds, hold for 4, exhale for 4, hold for 4. Repeat 4 times.",
                "triggers": ["high_stress", "anxiety", "overwhelm"]
            },
            {
                "id": "visual_reset",
                "name": "20-20-20 Rule",
                "description": "Look at something 20 feet away for 20 seconds every 20 minutes",
                "duration_seconds": 20,
                "effectiveness_stress": 0.4,
                "effectiveness_focus": 0.8,
                "category": "eye_rest",
                "instructions": "Look away from your screen at something 20 feet (6 meters) away for 20 seconds.",
                "triggers": ["eye_strain", "declining_focus", "long_session"]
            },
            {
                "id": "micro_stretch",
                "name": "Desk Micro-Stretch",
                "description": "Quick 30-second stretches for desk workers",
                "duration_seconds": 30,
                "effectiveness_stress": 0.6,
                "effectiveness_focus": 0.5,
                "category": "physical",
                "instructions": "Neck rolls (10s), shoulder shrugs (10s), wrist rotations (10s).",
                "triggers": ["poor_posture", "physical_tension", "fatigue"]
            },
            {
                "id": "mindful_pause",
                "name": "Mindful Pause",
                "description": "Brief mindfulness check-in",
                "duration_seconds": 45,
                "effectiveness_stress": 0.7,
                "effectiveness_focus": 0.7,
                "category": "mindfulness",
                "instructions": "Close eyes. Notice 3 sounds, 3 sensations, 3 breaths. Open eyes.",
                "triggers": ["mental_fog", "scattered_attention", "emotional_overwhelm"]
            },
            {
                "id": "power_posture",
                "name": "Power Posture Reset",
                "description": "Adjust posture for confidence and alertness",
                "duration_seconds": 15,
                "effectiveness_stress": 0.5,
                "effectiveness_focus": 0.6,
                "category": "posture",
                "instructions": "Sit tall, shoulders back, feet flat. Take 3 deep breaths.",
                "triggers": ["slouching", "low_energy", "confidence_drop"]
            },
            {
                "id": "gratitude_moment",
                "name": "Quick Gratitude",
                "description": "Think of one thing you're grateful for",
                "duration_seconds": 20,
                "effectiveness_stress": 0.6,
                "effectiveness_focus": 0.4,
                "category": "emotional",
                "instructions": "Think of one specific thing you're grateful for right now. Smile.",
                "triggers": ["negativity", "stress", "frustration"]
            },
            {
                "id": "cold_water",
                "name": "Cold Water Refresh",
                "description": "Drink cold water to boost alertness",
                "duration_seconds": 10,
                "effectiveness_stress": 0.3,
                "effectiveness_focus": 0.7,
                "category": "hydration",
                "instructions": "Drink a glass of cold water slowly. Feel the temperature.",
                "triggers": ["drowsiness", "mental_fog", "low_focus"]
            },
            {
                "id": "desk_yoga",
                "name": "Seated Spinal Twist",
                "description": "Gentle twist to release tension",
                "duration_seconds": 40,
                "effectiveness_stress": 0.7,
                "effectiveness_focus": 0.5,
                "category": "physical",
                "instructions": "Sit tall, place right hand on left knee, twist left. Hold 20s. Repeat other side.",
                "triggers": ["back_tension", "stiffness", "stress"]
            },
            {
                "id": "pomodoro_break",
                "name": "Pomodoro Micro-Break",
                "description": "Classic 5-minute break after focused work",
                "duration_seconds": 300,
                "effectiveness_stress": 0.6,
                "effectiveness_focus": 0.8,
                "category": "break",
                "instructions": "Step away from screen. Walk, stretch, or look outside. No screens.",
                "triggers": ["long_focus_session", "mental_fatigue", "25min_timer"]
            },
            {
                "id": "progressive_relaxation",
                "name": "Quick Progressive Relaxation",
                "description": "Tense and release muscle groups",
                "duration_seconds": 90,
                "effectiveness_stress": 0.85,
                "effectiveness_focus": 0.5,
                "category": "relaxation",
                "instructions": "Tense face (5s), release. Tense shoulders (5s), release. Tense hands (5s), release. Repeat.",
                "triggers": ["high_stress", "physical_tension", "anxiety"]
            }
        ]
    
    def _build_index(self):
        """Build FAISS index for fast retrieval"""
        # Create text descriptions for embedding
        texts = []
        for intervention in self.interventions:
            text = f"{intervention['name']}. {intervention['description']}. "
            text += f"Category: {intervention['category']}. "
            text += f"Triggers: {', '.join(intervention['triggers'])}."
            texts.append(text)
        
        # Generate embeddings
        print("Generating embeddings for interventions...")
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
        self.index.add(embeddings)
        
        print(f"Built FAISS index with {len(self.interventions)} interventions")
    
    def retrieve_interventions(self, state, top_k=3):
        """
        Retrieve best interventions for current mental state
        
        Args:
            state: dict with keys like 'stress', 'focus', 'fatigue', etc.
            top_k: number of interventions to return
        
        Returns:
            list of intervention dicts with scores
        """
        # Create query text based on state
        query_parts = []
        
        if state.get('stress', 0) > 0.6:
            query_parts.append("high stress anxiety overwhelm")
        if state.get('focus', 1) < 0.4:
            query_parts.append("low focus distracted scattered attention")
        if state.get('fatigue', 0) > 0.6:
            query_parts.append("tired fatigue drowsy mental fog")
        if state.get('eye_strain', False):
            query_parts.append("eye strain visual fatigue")
        if state.get('physical_tension', False):
            query_parts.append("tension stiff physical discomfort")
        
        if not query_parts:
            query_parts.append("maintain focus general wellbeing")
        
        query_text = " ".join(query_parts)
        
        # Encode query
        query_embedding = self.encoder.encode([query_text], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Rank by effectiveness for current state
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            intervention = self.interventions[idx].copy()
            
            # Calculate state-specific effectiveness score
            effectiveness = 0.0
            if state.get('stress', 0) > 0.5:
                effectiveness += intervention['effectiveness_stress'] * state['stress']
            if state.get('focus', 1) < 0.5:
                effectiveness += intervention['effectiveness_focus'] * (1 - state['focus'])
            
            # Combine semantic similarity with effectiveness
            final_score = 0.6 * float(similarity) + 0.4 * effectiveness
            
            intervention['retrieval_score'] = final_score
            intervention['semantic_similarity'] = float(similarity)
            results.append(intervention)
        
        # Sort by final score
        results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        
        return results
    
    def get_intervention_by_id(self, intervention_id):
        """Get specific intervention by ID"""
        for intervention in self.interventions:
            if intervention['id'] == intervention_id:
                return intervention
        return None


def generate_intervention_text(intervention, llm_model=None):
    """
    Generate natural language intervention using Hugging Face model
    Falls back to template if model not available
    """
    if llm_model is None:
        # Template-based generation
        return f"💡 **{intervention['name']}**\n\n" \
               f"{intervention['description']}\n\n" \
               f"⏱️ Duration: {intervention['duration_seconds']} seconds\n\n" \
               f"📋 Instructions:\n{intervention['instructions']}"
    
    # TODO: Use actual Hugging Face model for more natural generation
    # Example: phi-2, distilgpt2, or mistral-7b-instruct
    # For now, return template
    return f"💡 **{intervention['name']}**\n\n" \
           f"{intervention['description']}\n\n" \
           f"⏱️ Duration: {intervention['duration_seconds']} seconds\n\n" \
           f"📋 Instructions:\n{intervention['instructions']}"


if __name__ == '__main__':
    # Test retrieval system
    print("Initializing Intervention Retriever...")
    retriever = InterventionRetriever()
    
    # Test different states
    test_states = [
        {
            'name': 'High Stress',
            'state': {'stress': 0.85, 'focus': 0.6, 'fatigue': 0.4}
        },
        {
            'name': 'Low Focus',
            'state': {'stress': 0.3, 'focus': 0.25, 'fatigue': 0.6}
        },
        {
            'name': 'Eye Strain',
            'state': {'stress': 0.4, 'focus': 0.5, 'eye_strain': True}
        }
    ]
    
    for test in test_states:
        print(f"\n{'='*60}")
        print(f"Testing: {test['name']}")
        print(f"State: {test['state']}")
        print(f"{'='*60}")
        
        interventions = retriever.retrieve_interventions(test['state'], top_k=3)
        
        for i, intervention in enumerate(interventions, 1):
            print(f"\n{i}. {intervention['name']} (Score: {intervention['retrieval_score']:.3f})")
            print(f"   Category: {intervention['category']}")
            print(f"   Duration: {intervention['duration_seconds']}s")
            print(f"   Semantic Similarity: {intervention['semantic_similarity']:.3f}")
            print(f"   Instructions: {intervention['instructions'][:80]}...")