# MergeKit for MLX

[![PyPI version](https://badge.fury.io/py/mlx-lm.svg)](https://badge.fury.io/py/mlx-lm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

MergeKit for MLX is an advanced model merging toolkit that enables the combination of language models with different architectures and layer counts. This extension to MLX-LM brings the powerful merging capabilities of MergeKit to Apple Silicon devices, optimized for MLX's array framework.

## Key Features

- **Cross-Architecture Merging**: Combine models with different sizes and architectures (e.g., QwQ-32B with 64 layers and Bielik with 40 layers)
- **Multiple Merging Algorithms**: Support for SLERP, Linear, TIES, DARE, and Passthrough methods
- **Slice-Based Approach**: Apply different merging strategies to different parts of the models
- **Layer-Level Customization**: Fine-grained control over which layers to merge and how
- **Memory Efficiency**: Optimized for Apple Silicon's unified memory architecture
- **Simple Configuration**: Intuitive YAML configuration format

## Installation

MergeKit is included in the MLX-LM package. To install:

```bash
pip install mlx-lm
# Or for the development version
pip install git+https://github.com/ml-explore/mlx-examples.git#subdirectory=llms/mlx_lm
```

## Quick Start

### Basic Usage

```bash
mlx_lm.mergekit --config your_config.yaml --mlx-path output_model --verbose
```

### Example Configuration

This example merges the first 40 layers of QwQ-32B with Bielik using SLERP, while preserving the remaining layers of QwQ:

```yaml
models:
  - name: qwq
    path: Qwen/QwQ-32B
    base_model: true
  - name: bielik
    path: speakleash/Bielik-11B-v2.3-Instruct

slices:
  - sources:
      - model: qwq
        layer_range: [0, 40]
      - model: bielik
        layer_range: [0, 40]
    merge_method: slerp
    parameters:
      t:
        - filter: self_attn
          value: [0, 0.3, 0.6, 0.8, 1]
        - filter: mlp
          value: [1, 0.7, 0.4, 0.2, 0]
        - filter: embed
          value: 0
        - value: 0.5
  - sources:
      - model: qwq
        layer_range: [40, 64]
    merge_method: passthrough

tokenizer:
  source: qwq
  preserve_embeddings: true
```

### Using the Merged Model

```bash
# Generate text with the merged model
mlx_lm.generate --model output_model --prompt "Your prompt here" --max-tokens 500
```

## Detailed Documentation

### Configuration Format

The YAML configuration file consists of four main sections:

#### 1. Models

Define the models to be merged:

```yaml
models:
  - name: model1  # A unique identifier
    path: path/to/model1  # Local path or Hugging Face model ID
    base_model: true  # Mark as the base model (required for one model)
  - name: model2
    path: path/to/model2
```

#### 2. Slices

Define how different parts of the models should be merged:

```yaml
slices:
  - sources:  # List of model parts to merge in this slice
      - model: model1  # Model name from models section
        layer_range: [0, 12]  # [start, end) range of layers
      - model: model2
        layer_range: [0, 12]
    merge_method: slerp  # Merging algorithm to use
    parameters:  # Method-specific parameters
      t:  # For SLERP/Linear/TIES/DARE
        - filter: self_attn  # Apply to specific parameter types
          value: [0, 0.5]  # Values or range of values
        - filter: mlp
          value: [1, 0.5]
        - value: 0.5  # Default value for other parameters
```

#### 3. Tokenizer

Specify which model's tokenizer to use:

```yaml
tokenizer:
  source: model1  # Name from models section
  preserve_embeddings: true  # Keep original embeddings
```

#### 4. Advanced Options (Optional)

Additional configuration options:

```yaml
advanced:
  low_memory: true  # Enable memory-saving optimizations
  dtype: bfloat16  # Set computation precision
```

### Merging Methods

MergeKit supports the following merging methods:

#### SLERP (Spherical Linear Interpolation)

```yaml
merge_method: slerp
```

Interpolates between weights on a spherical path, which often preserves model quality better than linear interpolation. Recommended for most use cases.

#### Linear

```yaml
merge_method: linear
```

Simple linear interpolation: `(1-t) * w1 + t * w2`. Fast but may result in lower quality than SLERP.

#### TIES (Task-Specific Interpolation of Embeddings and Weights)

```yaml
merge_method: ties
```

Preserves the signs of weights from the dominant model while using magnitudes from the interpolated weights. Helps maintain task-specific knowledge from both models.

#### DARE (Drop and REscale)

```yaml
merge_method: dare
parameters:
  density: 0.4  # Controls sparsity (0.0-1.0)
  t:
    - value: 0.5
```

Prunes weights below a certain magnitude threshold after interpolation, then rescales the remaining weights. Useful for reducing model size while maintaining performance.

#### Passthrough

```yaml
merge_method: passthrough
```

Simply copies weights from a source model without merging. Used for preserving unique layers or for layers where merging isn't possible.

### Parameter Filters

You can apply different merging parameters to different parts of the model using filters:

```yaml
parameters:
  t:
    - filter: self_attn  # Applies to attention weights
      value: 0.2
    - filter: mlp  # Applies to MLP weights
      value: 0.8
    - value: 0.5  # Default for everything else
```

Common filters include:
- `self_attn`: Self-attention layers
- `mlp`: MLP/feed-forward layers
- `embed`: Embedding layers
- `head`: Output head
- `norm`: Normalization layers

### Parameter Gradients

You can specify gradients of values across layers:

```yaml
value: [0, 0.5, 0.3, 0.7, 1]
```

This creates a gradient of values across the layers in the specified range.

## Advanced Usage

### Memory Optimization

For large models on memory-constrained devices:

```yaml
advanced:
  low_memory: true
  offload_to_cpu: true
  batch_size: 1
```

### Python API

```python
import mlx_lm

# Merge models using a configuration file
mlx_lm.mergekit(
    config="your_config.yaml",
    mlx_path="output_model",
    verbose=True
)

# Use the merged model
model, tokenizer = mlx_lm.load("output_model")
tokens = mlx_lm.generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Your prompt here",
    max_tokens=500,
    temp=0.7
)
print(tokenizer.decode(tokens)[0])
```

### Creating Multi-Lingual Models

To create a model that combines strengths from models trained on different languages:

```yaml
models:
  - name: base_model
    path: mistralai/Mistral-7B-v0.1
    base_model: true
  - name: polish_model
    path: speakleash/Bielik-11B-v2.3-Instruct

slices:
  - sources:
      - model: base_model
        layer_range: [0, 32]
      - model: polish_model
        layer_range: [0, 32]
    merge_method: slerp
    parameters:
      t:
        - filter: self_attn
          value: [0.2, 0.4, 0.6]
        - filter: mlp
          value: [0.5, 0.3, 0.1]
        - value: 0.3

tokenizer:
  source: polish_model
  preserve_embeddings: true
```

## Examples

### Merging QwQ-32B with Bielik (Polish LLM)

See the [Quick Start](#quick-start) example.

### Creating a Math-Specialized Model

```yaml
models:
  - name: base
    path: mistralai/Mistral-7B-v0.1
    base_model: true
  - name: math
    path: microsoft/Phi-2

slices:
  - sources:
      - model: base
        layer_range: [0, 32]
      - model: math
        layer_range: [0, 32]
    merge_method: ties
    parameters:
      t:
        - filter: mlp
          value: 0.7
        - value: 0.3
```

### Merging Models with Different Sizes

```yaml
models:
  - name: large
    path: meta-llama/Llama-2-13b
    base_model: true
  - name: small
    path: TinyLlama/TinyLlama-1.1B

slices:
  # Merge the first 12 layers
  - sources:
      - model: large
        layer_range: [0, 12]
      - model: small
        layer_range: [0, 6]  # Repeated to match large model
    merge_method: slerp
    parameters:
      t:
        - value: 0.3
  # Keep remaining layers from large model
  - sources:
      - model: large
        layer_range: [12, 40]
    merge_method: passthrough
```

## Troubleshooting

### Out of Memory Errors

If you encounter memory issues:
1. Use a more aggressive quantization (4-bit)
2. Enable `low_memory` mode in advanced options
3. Reduce batch_size to 1
4. Try merging in smaller slices

### Incompatible Layer Dimensions

If layer dimensions don't match:
1. Check if models have compatible architectures
2. Try using the `passthrough` method for incompatible layers
3. Consider adding an adapter layer between models

## Contributing

Contributions to MergeKit for MLX are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes

Please ensure your code follows the project's style and passes all tests.

## License

MergeKit for MLX is released under the Apache 2.0 License.

## Acknowledgments

MergeKit for MLX is inspired by:
- [Arcee's MergeKit](https://github.com/arcee-ai/mergekit)
- [MLX framework](https://github.com/ml-explore/mlx) by Apple
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)

## Citation

If you use MergeKit for MLX in your research, please cite:

```bibtex
@software{mergekit_mlx,
  author = {Contributors},
  title = {MergeKit for MLX: Advanced Model Merging on Apple Silicon},
  url = {https://github.com/ml-explore/mlx-examples},
  year = {2025},
}
```
