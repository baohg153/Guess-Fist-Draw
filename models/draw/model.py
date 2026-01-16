import torch.nn as nn

class DrawClassification(nn.Module):
    def __init__(self, input_size=28, num_classes=30)->None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        flatten_size = 32 * (input_size // 4) * (input_size // 4)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input):
        output = self.conv(input)
        output = self.linear(output)
        return output