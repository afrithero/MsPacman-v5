import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_FLAT_DIM = 22 * 16 * 64 

class AtariNetDQN(nn.Module):
    def __init__(self, num_actions=4, init_weights=True, dueling=False):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4), 
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )

        self.dueling = dueling
        
        if self.dueling:
            self.value = nn.Sequential(
                nn.Linear(_FLAT_DIM, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(_FLAT_DIM, 512), nn.ReLU(inplace=True),
                nn.Linear(512, num_actions),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(_FLAT_DIM, 512), nn.ReLU(inplace=True),
                nn.Linear(512, num_actions),
            )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_nchw_float01(x)

        f = self.cnn(x)                 # (N, 64, 22, 16)
        f = torch.flatten(f, start_dim=1)  # (N, 64*22*16)

        if self.dueling:
            V = self.value(f)                         # (N,1)
            A = self.advantage(f)                     # (N,num_actions)
            return V + (A - A.mean(dim=1, keepdim=True))
        else:
            return self.classifier(f)
            
    def _ensure_nchw_float01(self, x):
        x = torch.tensor(x, dtype=torch.float).cuda()

        if x.dim() == 3:  # (H,W,3) or (3,H,W)
            if x.shape[0] == 3:  # (3,H,W)
                x = x.unsqueeze(0)
            else:                # (H,W,3)
                x = x.unsqueeze(0).permute(0, 3, 1, 2)
        elif x.dim() == 4 and x.shape[1] != 3:  # (N,H,W,3)
            x = x.permute(0, 3, 1, 2)

        x = x / 255.0

        return x

