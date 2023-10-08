import torch.nn as nn
from typing import List 
from torchvision import models
from torchvision import transforms
    
class ResNetWithHooks(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.output_layers_extracted = {}
        self.pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.fhooks = []

        # attach hooks for requested output layers
        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if l in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))

    # attaching hooks to extract output layers    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.output_layers_extracted[layer_name] = output
        return hook

    # used internally by the model
    def forward(self, x):
        out = self.pretrained(x)
        return out, self.output_layers_extracted

    # fetches dictionary of requested output layers
    def get_output_layers(self):
        return self.output_layers_extracted
    
    # preprocess images before processing them using ResNet50
    def preprocess(self,image):
        tfs = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transformed_image = tfs(image)
        self.output_layers_extracted = {}
        return transformed_image.unsqueeze(0)

