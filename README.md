Wrapper function for nni pruning method


| Parameter            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `model`              | The PyTorch model to be pruned.                                             |
| `sparse_ratio`       | The ratio of sparsity to be applied to the model.                           |
| `input_shape`        | The shape of the input tensor that the model expects.                       |
| `pruned_layer_types` | A list of layer types to be considered for pruning    (default: `['Linear''Conv2d',   'Conv3d', 'BatchNorm2d']`)          |
| `exclude_layer_names`| A list of layer names to be excluded from pruning  (default: `None`, will automatically detect it, could possibly cause an error)|
| `prunner_choice`     | The choice of pruner to be used   (default: `None`, will select L1NormPrunner).   |





Returns 

| Parameter            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `pruned_model`              | The pruned PyTorch model.                                             |




Note that when loading the model, the relative path to the model definition has to be the same as when the model was first created.
possible solution.
ONNX FORMAT
