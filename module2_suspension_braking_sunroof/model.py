"""Model configs for Module 2 — CNN Terrain, RF Gear/Sunroof, MLP Suspension."""
from __future__ import annotations
import torch.nn as nn
from ..common.torch_helper import MLPRegressor

# =======================================================
# 1. CNN for Terrain Classification (Vision Pipeline)
# =======================================================
class TerrainCNN(nn.Module):
    """Processes front camera frames to classify terrain: 0=Plains, 1=Desert, 2=Rocky, 3=Extreme."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 4) # 4 terrain classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =======================================================
# 2. MLP for Continuous Suspension Control
# =======================================================
MLP_FEATURES = ["terrain_type", "speed", "load_factor", "engine_heat"]
MLP_TARGETS = ["k_eff", "c_eff", "heater_power"]

def build_suspension_mlp():
    return MLPRegressor(n_features=len(MLP_FEATURES), n_outputs=len(MLP_TARGETS), hidden=(128, 64))

# =======================================================
# 3. Random Forest Feature Maps (For Gear & Sunroof)
# =======================================================
# Used by sklearn RandomForestClassifier in train.py
RF_GEAR_FEATURES = ["speed", "load_factor", "throttle_direction"]
RF_GEAR_TARGETS = ["target_gear"] # 0=Reverse, 1-6=Forward

RF_SUNROOF_FEATURES = ["outside_temp", "cabin_temp", "humidity", "rain", "sunlight", "aqi", "speed", "passenger_pref"]
RF_SUNROOF_TARGETS = ["sunroof_state"] # 1=Open, 0=Closed
