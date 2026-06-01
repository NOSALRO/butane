import unittest
import torch
import torch.nn as nn
import os
import tempfile
import butane

class SimpleConvNet(nn.Module):
    """A slightly larger model to test shapes and layers."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Weight + Bias
            nn.BatchNorm2d(16),                        # Running stats (should be ignored by EMA)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

class TestSwitchEMA(unittest.TestCase):
    def setUp(self):
        # Deterministic setup
        torch.manual_seed(42)
        self.model = SimpleConvNet()
        # High decay to make math visible, but not 1.0
        self.ema = butane.nn.EMA(self.model, decay=0.9) 
        self.input_dummy = torch.randn(2, 3, 8, 8)

    def test_update_mechanics(self):
        """Test that update() actually moves the shadow weights."""
        # 1. Store initial shadow
        initial_shadow = self.ema._shadow['classifier.weight'].clone()
        
        # 2. Modify model manually (simulate optimization step)
        with torch.no_grad():
            self.model.classifier.weight.add_(1.0)
        
        # 3. Update EMA
        self.ema.update()
        
        # 4. Check logic:
        # New Shadow = 0.9 * Old_Shadow + 0.1 * New_Param
        # Param increased by 1.0. Shadow should move slightly towards it.
        new_shadow = self.ema._shadow['classifier.weight']
        
        self.assertFalse(torch.equal(initial_shadow, new_shadow), "Shadow weights did not update.")
        
        # Verify lerp direction
        # Since param > initial_shadow, new_shadow should be > initial_shadow
        self.assertTrue(torch.all(new_shadow > initial_shadow), "EMA moved in wrong direction.")

    def test_switch_logic(self):
        """Test that apply_switch() overwrites the model."""
        # Diverge model and EMA significantly
        with torch.no_grad():
            self.model.classifier.weight.fill_(100.0)
            self.ema._shadow['classifier.weight'].fill_(0.0)
            
        # Apply Switch
        self.ema.switch()
        
        # Model should now be 0.0
        self.assertTrue(
            torch.allclose(self.model.classifier.weight, torch.zeros_like(self.model.classifier.weight)),
            "Switch failed: Model weights were not overwritten by EMA."
        )

    def test_exclude_bias(self):
        """Test that bias exclusion works to save memory/compute."""
        ema_bias = butane.nn.EMA(self.model, decay=0.9, exclude_bias=True)
        
        # Check if biases are in shadow dict
        for name in ema_bias._shadow.keys():
            self.assertFalse(name.endswith('.bias'), f"Bias {name} was not excluded!")
            
        # Verify Conv2d weights ARE tracked
        self.assertIn('features.0.weight', ema_bias._shadow)

    def test_save_load_consistency(self):
        """Test state_dict saving and loading on a fresh instance."""
        # 1. Modify State
        self.ema.update()
        
        # 2. Save
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(self.ema.state_dict(), f.name)
            path = f.name
            
        try:
            # 3. Create fresh model and EMA
            new_model = SimpleConvNet()
            new_ema = butane.nn.EMA(new_model, decay=0.9)
            
            # Verify they are different initially (due to random init of new_model)
            # Note: buffers init from model, so we need to check if loading the FILE restores the OLD state
            
            # 4. Load
            state_dict = torch.load(path)
            new_ema.load_state_dict(state_dict)
            
            # 5. Compare
            # Iterate over original shadow and new shadow
            for name, old_tensor in self.ema._shadow.items():
                new_tensor = new_ema._shadow[name]
                self.assertTrue(
                    torch.equal(old_tensor, new_tensor),
                    f"Mismatch in parameter {name} after loading."
                )
                
            # 6. Verify Optimization bindings are restored
            # If bindings are broken, update() might fail or do nothing
            new_ema.update() # Should not crash
            
        finally:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    unittest.main()
