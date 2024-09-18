import torch 
from models import myVGG
device ='cpu'
# To load the pruned model later
loaded_model = torch.load('pruned_model.pth').to(device)
print("Pruned model loaded from pruned_model.pth")
print(loaded_model)