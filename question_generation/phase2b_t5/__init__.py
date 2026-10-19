"""
Phase 2B — T5 fine-tuning for question generation.

Public API
----------
T5Trainer         : fine-tune T5-small on SQuAD question generation data
T5InferenceEngine : load a fine-tuned checkpoint and generate questions
T5QGDataset       : PyTorch Dataset that pre-tokenises T5 inputs/targets
"""

from phase2b_t5.trainer   import T5Trainer
from phase2b_t5.inference import T5InferenceEngine
from phase2b_t5.dataset   import T5QGDataset
