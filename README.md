# MLX-LM with MergeKit

This repository extends the [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) package with advanced model merging capabilities from [MergeKit](https://github.com/arcee-ai/mergekit), optimized for Apple Silicon.

## Features

- **Cross-Architecture Model Merging**: Combine models with different sizes and architectures
- **Multiple Merging Algorithms**: SLERP, Linear, TIES, DARE, and Passthrough
- **Layer-Level Customization**: Apply different merge strategies to different parts of the models
- **Memory Optimizations**: Special modes for large models on memory-constrained devices
- **MLX Optimizations**: Takes advantage of MLX's unified memory architecture for Apple Silicon

## Installation

```bash
# Clone the repository
git clone https://github.com/Szowesgad/mlx-lm.git
cd mlx-lm
git checkout mlx-mergekit

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Merge models
mergekit --config mergekit/examples/qwq_bielik_merge.yaml --mlx-path merged_model --verbose

# Test the merged model
python mergekit/tests/test_merge.py --config mergekit/examples/qwq_bielik_merge.yaml --output-dir merged_model --verbose
```

## Example: Merging QwQ-32B with Bielik

This example combines QwQ-32B's advanced reasoning capabilities with Bielik's Polish language knowledge:

```yaml
# mergekit/examples/qwq_bielik_merge.yaml
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

## Documentation

For detailed documentation, see:

- [MergeKit Documentation](mergekit/readme.md) - Complete user guide
- [Development Stage](mergekit/docs/devstage.md) - Current development status and roadmap

## Requirements

- Python 3.9+
- MLX 0.3.0+
- MLX-LM 0.2.0+
- Apple Silicon Mac (M1/M2/M3 series)
- 16GB+ RAM (48GB+ recommended for large models)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- [MLX by Apple](https://github.com/ml-explore/mlx)
- [MergeKit by Arcee](https://github.com/arcee-ai/mergekit)
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
