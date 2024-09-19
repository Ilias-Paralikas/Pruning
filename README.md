This work was partially support by the ``Trustworthy And Resilient Decentralised Intelligence For Edge Systems (TaRDIS)" Project, funded by EU HORIZON EUROPE program, under grant agreement No 101093006


The function to be exposed is import as follows

`from pruning import prune_model `

as used as such

Wrapper function for nni pruning method
| Parameter            | Type                | Description                                                                 |
|----------------------|---------------------|-----------------------------------------------------------------------------|
| `model`              | `torch.nn.Module`   | The PyTorch model to be pruned.                                             |
| `sparse_ratio`       | `float`             | The ratio of sparsity to be applied to the model.                           |
| `input_shape`        | `tuple`             | The shape of the input tensor that the model expects.                       |
| `pruned_layer_types` | `list`              | A list of layer types to be considered for pruning (default: `['Linear', 'Conv2d', 'Conv3d', 'BatchNorm2d']`). |
| `exclude_layer_names`| `list` or `None`    | A list of layer names to be excluded from pruning (default: `None`, will automatically detect it, could possibly cause an error). |
| `prunner_choice`     | `str` or `None`     | The choice of pruner to be used (default: `None`, will select `L1NormPruner`).                                |



Returns 

| Parameter            | Type                | Description                                                                 |
|----------------------|---------------------|-----------------------------------------------------------------------------|
| `model`              | `torch.nn.Module`   | The pruned PyTorch model                                         |



Note that when loading the model, the relative path to the model definition has to be the same as when the model was first created.
possible solution.
ONNX FORMAT
