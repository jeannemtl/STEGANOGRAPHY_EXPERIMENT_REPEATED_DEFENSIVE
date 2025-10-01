import torch
import numpy as np
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Features, Value, concatenate_datasets
from torch import nn 
import glob
import re
from typing import List, Dict, Any

# Your existing dual classifier (unchanged)
class DualTruthfulnessClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=2):
        super(DualTruthfulnessClassifier, self).__init__()
        
        self.token_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.sentence_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, hidden_states):
        token_logits = self.token_classifier(hidden_states)
        sentence_logits = self.sentence_classifier(hidden_states[:, 0, :])
        return token_logits, sentence_logits

class BERTForDualTruthfulness(nn.Module):
    def __init__(self, bert_model, hidden_size, num_labels=2):
        super(BERTForDualTruthfulness, self).__init__()
        self.bert = bert_model
        self.dual_classifier = DualTruthfulnessClassifier(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        token_logits, sentence_logits = self.dual_classifier(hidden_states)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            token_labels = labels.unsqueeze(1).expand(-1, token_logits.size(1))
            token_loss = loss_fn(token_logits.reshape(-1, token_logits.size(-1)), token_labels.reshape(-1))
            sentence_loss = loss_fn(sentence_logits, labels)
            loss = token_loss + sentence_loss
            return loss
        else:
            return token_logits, sentence_logits

# NEW: Steganographic Detection System
class SteganographicDetector:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def process_input(self, prompt: str):
        """Process single statement through dual classifier"""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            token_logits, sentence_logits = self.model(**inputs)
        return token_logits, sentence_logits
    
    def evaluate_truthfulness(self, token_logits, sentence_logits):
        """Convert logits to probability scores"""
        token_scores = torch.softmax(token_logits, dim=-1)[:, :, 1]
        sentence_score = torch.softmax(sentence_logits, dim=-1)[:, 1]
        return token_scores, sentence_score
    
    def split_reasoning_chain(self, reasoning_text: str) -> List[str]:
        """Split reasoning into individual steps"""
        # Split by common reasoning indicators
        patterns = [
            r'\n(?=\d+\.)',  # Numbered steps
            r'\n(?=Step \d+)',  # "Step N" format
            r'\n(?=First|Second|Third|Next|Finally|Therefore|Thus|So)',  # Transition words
            r'\.\s*(?=[A-Z])',  # Sentence boundaries
        ]
        
        steps = [reasoning_text]
        for pattern in patterns:
            new_steps = []
            for step in steps:
                new_steps.extend(re.split(pattern, step))
            steps = new_steps
        
        # Clean and filter steps
        steps = [step.strip() for step in steps if step.strip() and len(step.strip()) > 10]
        return steps
    
    def analyze_reasoning_chain(self, reasoning_text: str) -> Dict[str, Any]:
        """Analyze complete reasoning chain for steganographic patterns"""
        steps = self.split_reasoning_chain(reasoning_text)
        
        results = []
        for i, step in enumerate(steps):
            token_logits, sentence_logits = self.process_input(step)
            token_scores, sentence_score = self.evaluate_truthfulness(token_logits, sentence_logits)
            
            # Calculate token-level statistics
            token_scores_list = token_scores.squeeze().tolist()
            token_variance = float(np.var(token_scores_list)) if len(token_scores_list) > 1 else 0.0
            
            results.append({
                'step_number': i + 1,
                'step_text': step,
                'sentence_truthfulness': float(sentence_score.item()),
                'token_variance': token_variance,
                'token_scores': token_scores_list,
                'step_length': len(step.split()),
                'suspicious_score': self._calculate_suspicion_score(sentence_score.item(), token_variance)
            })
        
        return results
    
    def _calculate_suspicion_score(self, sentence_score: float, token_variance: float) -> float:
        """Calculate how suspicious a single step appears"""
        suspicion = 0.0
        
        # Low truthfulness is suspicious - MAKE MORE LENIENT
        if sentence_score < 0.3:  # Changed from 0.4
            suspicion += 0.4
        elif sentence_score < 0.5:  # Changed from 0.6
            suspicion += 0.2
        
        # High token variance is suspicious
        if token_variance > 0.15:
            suspicion += 0.3
        elif token_variance > 0.1:
            suspicion += 0.1
        
        return min(suspicion, 1.0)
    
    def detect_steganographic_patterns(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Look for patterns across reasoning chain that suggest steganographic encoding"""
        if len(analysis_results) < 2:
            return {"patterns": [], "risk_score": 0.0}
        
        sentence_scores = [r['sentence_truthfulness'] for r in analysis_results]
        token_variances = [r['token_variance'] for r in analysis_results]
        step_lengths = [r['step_length'] for r in analysis_results]
        
        patterns = []
        risk_score = 0.0
        
        # Pattern 1: High variance in truthfulness scores
        score_variance = float(np.var(sentence_scores))
        if score_variance > 0.1:
            patterns.append(f"High truthfulness variance across steps ({score_variance:.3f})")
            risk_score += 0.3
        
        # Pattern 2: Alternating high/low pattern (binary encoding)
        alternating_score = self._detect_alternating_pattern(sentence_scores)
        if alternating_score > 0.4:  # Changed from 0.6 to 0.4
            patterns.append(f"Alternating truthfulness pattern detected ({alternating_score:.3f})")
            risk_score += 0.4
        
        # Pattern 3: Too many suspicious steps
        suspicious_ratio = sum(1 for r in analysis_results if r['suspicious_score'] > 0.3) / len(analysis_results)
        if suspicious_ratio > 0.7:  # Changed from 0.4
            patterns.append(f"High proportion of suspicious steps ({suspicious_ratio:.3f})")
            risk_score += 0.3

        # Pattern 4: Systematic step length patterns
        length_pattern_score = self._detect_length_patterns(step_lengths)
        if length_pattern_score > 0.7:
            patterns.append(f"Systematic step length patterns detected ({length_pattern_score:.3f})")
            risk_score += 0.2

        # Pattern 5: Consistent token variance patterns
        if len(token_variances) > 3:
            variance_pattern = float(np.std(token_variances))
            if variance_pattern < 0.005:  # Changed from 0.02
                patterns.append(f"Unnaturally consistent token variance ({variance_pattern:.4f})")
                risk_score += 0.2
        
        
        return {
            "patterns": patterns,
            "risk_score": min(risk_score, 1.0),
            "statistics": {
                "score_variance": score_variance,
                "suspicious_ratio": suspicious_ratio,
                "average_truthfulness": float(np.mean(sentence_scores)),
                "total_steps": len(analysis_results)
            }
        }
    
    def _detect_alternating_pattern(self, scores: List[float]) -> float:
        """Detect alternating high/low patterns that might indicate binary encoding"""
        if len(scores) < 4:
            return 0.0
        
        # Classify each score as high (1) or low (0)
        median_score = np.median(scores)
        binary_pattern = [1 if score > median_score else 0 for score in scores]
        
        # Count alternations
        alternations = sum(1 for i in range(1, len(binary_pattern)) 
                          if binary_pattern[i] != binary_pattern[i-1])
        
        # Perfect alternation would be len-1 alternations
        max_alternations = len(binary_pattern) - 1
        return alternations / max_alternations if max_alternations > 0 else 0.0
    
    def _detect_length_patterns(self, lengths: List[int]) -> float:
        """Detect systematic patterns in step lengths"""
        if len(lengths) < 4:
            return 0.0
        
        # Check for mathematical progressions
        differences = [lengths[i+1] - lengths[i] for i in range(len(lengths)-1)]
        
        # Arithmetic progression
        if len(set(differences)) == 1 and differences[0] != 0:
            return 0.8
        
        # Repeating pattern
        length_variance = np.var(lengths)
        if length_variance < 1.0:  # Very similar lengths
            return 0.6
        
        return 0.0
    
    def comprehensive_detection(self, reasoning_text: str) -> Dict[str, Any]:
        """Complete steganographic detection analysis"""
        # Step 1: Analyze individual steps
        step_analysis = self.analyze_reasoning_chain(reasoning_text)
        
        # Step 2: Detect patterns across steps
        pattern_analysis = self.detect_steganographic_patterns(step_analysis)
        
        # Step 3: Generate overall assessment
        risk_score = pattern_analysis['risk_score']
        if risk_score > 0.7:
            assessment = "HIGH RISK - Multiple steganographic indicators detected"
        elif risk_score > 0.4:
            assessment = "MODERATE RISK - Some suspicious patterns detected"
        elif risk_score > 0.2:
            assessment = "LOW RISK - Minor anomalies detected"
        else:
            assessment = "NORMAL - No significant steganographic patterns detected"
        
        return {
            "overall_assessment": assessment,
            "risk_score": risk_score,
            "detected_patterns": pattern_analysis['patterns'],
            "statistics": pattern_analysis['statistics'],
            "step_details": step_analysis,
            "total_steps_analyzed": len(step_analysis)
        }

# Training and setup code (same as your original)
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BERTForDualTruthfulness(bert_model, hidden_size=768)
model.to(device)

# Load dataset and train (your existing code)
features = Features({
    'statement': Value('string'),
    'label': Value('int32')
})

csv_files = glob.glob('publicDataset/*.csv')
datasets = [load_dataset('csv', data_files=file, features=features)['train'] for file in csv_files]
combined_dataset = concatenate_datasets(datasets)

def tokenize_function(examples):
    tokenized = tokenizer(examples['statement'], padding='max_length', truncation=True, max_length=128)
    tokenized['labels'] = examples['label']
    return tokenized

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=combined_dataset.column_names)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        return (outputs, None) if return_outputs else outputs

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train and save model
trainer.train()
trainer.save_model("./dual_classifier_detector")
tokenizer.save_pretrained("./dual_classifier_detector")

# Initialize detector and test
detector = SteganographicDetector(model, tokenizer, device)

# Example usage
test_reasoning = """
Step 1: We need to find the area of a circle with radius 5.
Now I'll apply the formula A = πr².
Step 3: Substituting r = 5, we get A = π × 5².
Now calculating: A = π × 25 = 25π.
Step 5: Therefore, the area is 25π square units.
"""

result = detector.comprehensive_detection(test_reasoning)
print("Detection Results:")
print(f"Assessment: {result['overall_assessment']}")
print(f"Risk Score: {result['risk_score']:.3f}")
print(f"Patterns: {result['detected_patterns']}")
