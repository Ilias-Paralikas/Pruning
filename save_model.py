import torch
from pruning import prune_model
from models import myVGG
            



SPARSE_RATIO = 0.5
INPUT_SHAPE = (1, 3, 32, 32) # including batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myVGG().to(device)
new_model = prune_model(model,SPARSE_RATIO,INPUT_SHAPE)
print(new_model)
print(model is new_model)


torch.save(new_model, 'pruned_model.pth')
print("Pruned model saved to pruned_model.pth")

'''

# Set the model to evaluation mode
model.eval()

# Define a dummy input tensor with the same shape as your input data
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# Export the model to ONNX format
torch.onnx.export(
    model,                # Model to be exported
    dummy_input,          # Dummy input tensor
    "model.onnx",         # Output file name
    export_params=True,   # Store the trained parameter weights inside the model file
    opset_version=11,     # ONNX version to export the model to
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=['input'],     # Input tensor names
    output_names=['output'],   # Output tensor names
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes
)

print("Model has been exported to model.onnx")
'''
