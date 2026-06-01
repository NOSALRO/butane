import os
import shutil
import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from unittest.mock import patch

# ==========================================
# IMPORT YOUR ACTUAL LOGGER HERE
# ==========================================
from butane.logger import ModelLogger

# A tiny model to satisfy the required `model` argument in load_checkpoint/checkpoint
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

def run_full_lifecycle_test():
    test_dir = "./test_wandb_lifecycle"
    model = DummyModel()
        
    print("\n🚀 [PHASE 1] Initial Baseline (overwrite=True, folder doesn't exist yet)")
    # We mock 'y' just in case, but the prompt shouldn't trigger since the folder is missing.
    logger1 = ModelLogger(fpath=test_dir, overwrite=True, resume=False, eval_mode=False, use_wandb=True)
    
    for step in range(1, 4): # Steps 1, 2, 3
        logger1.add_stats(loss=10.0 / step)
        logger1.checkpoint(model=model)
        logger1.commit()
        time.sleep(1)
        
    wandb.finish() 
    
    print("\n🌿 [PHASE 2] Branching & Stats Replay (overwrite=False, resume=True)")
    # This will load Step 3, rename the run, and replay Steps 1-3 into the new WandB dashboard!
    logger2 = ModelLogger(fpath=test_dir, overwrite=False, resume=True, eval_mode=False, use_wandb=True)
    logger2.load_checkpoint(step=3, model=model) 
    
    for step in range(4, 6): # Steps 4, 5
        logger2.add_stats(loss=5.0 / step)
        logger2.checkpoint(model=model)
        logger2.commit()
        time.sleep(1)
        
    wandb.finish()

    print("\n🔍 [PHASE 3] Eval Mode (eval_mode=True)")
    # Validating that no timestamp folders are created and WandB stays offline.
    logger3 = ModelLogger(fpath=test_dir, overwrite=False, resume=False, eval_mode=True, use_wandb=True)
    
    assert logger3.fpath.name == Path(test_dir).name, "Eval mode accidentally created a timestamped folder!"
    assert not getattr(logger3, '_use_wandb', True), "Eval mode failed to disable WandB!"
    print("✅ Eval mode successfully blocked timestamping and WandB init.")

    print("\n🔥 [PHASE 4] Hard Overwrite & Deletion (overwrite=True, resume=True)")
    # This WILL trigger the prompt because the folder exists. The mock will answer 'y'.
    # It should delete Phase 1 and Phase 2 from the server, then start fresh from Step 5!
    with patch('builtins.input', return_value='y'):
        logger4 = ModelLogger(fpath=test_dir, overwrite=True, resume=True, eval_mode=False, use_wandb=True)
    
    logger4.load_checkpoint(step=5, model=model)
    
    for step in range(6, 8): # Steps 6, 7
        logger4.add_stats(loss=0.1)
        logger4.checkpoint(model=model)
        logger4.commit()
        time.sleep(1)
        
    wandb.finish()
    print("\n✅ All tests complete! The matrix is bulletproof.")

if __name__ == "__main__":
    assert "WANDB_PROJECT" in os.environ, "Please set WANDB_PROJECT before running."
    run_full_lifecycle_test()
