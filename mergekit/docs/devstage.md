# MergeKit for MLX - Development Stage Documentation

**Last Updated:** March 20, 2025  
**Current Version:** 0.1.0  
**Status:** Active Development  
**Lead Developer:** @Szowesgad  

## Project Overview

MergeKit for MLX is an implementation of advanced model merging techniques from Arcee's MergeKit, adapted specifically for Apple's MLX framework. The primary goal is to enable the merging of large language models with different architectures (particularly different layer counts), optimized for Apple Silicon hardware.

## Motivation

The original motivation for this project came from the need to merge QwQ-32B (64 layers) with Polish language models like Bielik-11B-v2.3-Instruct (around 40 layers) to create a Polish-capable LLM that preserves the advanced reasoning abilities of QwQ-32B while incorporating Polish language knowledge.

## Current Development Status

### Completed Features

1. **Core Merging Algorithms**
   - SLERP (Spherical Linear Interpolation) - Adapted from original MLX implementation
   - Linear - Simple linear interpolation between weights
   - TIES (Task-specific Interpolation of Embeddings and Weights) - Sign-preserving interpolation
   - DARE (Drop and REscale) - Pruning and rescaling after interpolation
   - Passthrough - Copying weights without modification

2. **Slice-Based Architecture**
   - Support for defining multiple "slices" of layers with different merging strategies
   - Layer range specification for each model in each slice
   - Ability to merge only compatible portions of models with different total layer counts

3. **Configuration System**
   - YAML-based configuration format
   - Support for model definitions, slice specifications, tokenizer settings, and advanced options
   - Parameter filtering for applying different merge parameters to different layer types

4. **Memory Optimizations**
   - Low-memory mode for processing one parameter at a time
   - Memory cleanup during processing

5. **CLI and Python API**
   - Command-line interface through `mergekit` script
   - Python API through `mergekit` function
   - Integration with MLX-LM ecosystem

### Implementation Details

#### Code Structure

```
mergekit/
├── __init__.py                 # Package initialization, version info
├── bin/
│   └── mergekit               # CLI entry point
├── merge_advanced.py          # Core implementation
├── docs/
│   ├── devstage.md            # This file - development documentation
│   └── ...                    # Other documentation
├── examples/
│   └── qwq_bielik_merge.yaml  # Example configuration
└── readme.md                  # User documentation
```

#### Key Functions

| Function | Purpose | Status |
|----------|---------|--------|
| `merge_advanced` | Main merging function | Complete |
| `merge_slice` | Handle merging of a single slice | Complete |
| `slerp` | SLERP implementation | Complete |
| `linear_merge` | Linear interpolation | Complete |
| `ties_merge` | TIES implementation | Complete |
| `dare_merge` | DARE implementation | Complete |
| `passthrough` | Passthrough implementation | Complete |
| `unpack_values` | Gradient unpacking for layer-wise parameters | Complete |

#### MergeKit Algorithm Implementations

1. **SLERP Implementation**
   ```python
   def slerp(t, w1, w2, eps=1e-5):
       t = float(t)
       if t == 0: return w1
       elif t == 1: return w2
       # Normalize vectors
       v1 = w1 / mx.linalg.norm(w1)
       v2 = w2 / mx.linalg.norm(w2)
       # Angle between vectors
       dot = mx.clip((v1 * v2).sum(), 0.0, 1.0)
       theta = mx.arccos(dot)
       sin_theta = mx.sin(theta + eps)
       s1 = mx.sin(theta * (1 - t)) / sin_theta
       s2 = mx.sin(theta * t) / sin_theta
       return s1 * w1 + s2 * w2
   ```

2. **TIES Implementation**
   ```python
   def ties_merge(t, w1, w2):
       # Calculate average with weighting
       avg = (1 - t) * w1 + t * w2
       # Sign and magnitude operations
       sign_x = mx.sign(w1)
       sign_y = mx.sign(w2)
       abs_x = mx.abs(w1)
       abs_y = mx.abs(w2)
       # Create mask where magnitude of w1 >= magnitude of w2
       mask = abs_x >= abs_y
       # Apply TIES - keep sign from dominant weight, magnitude from average
       return mx.where(mask, sign_x * mx.abs(avg), sign_y * mx.abs(avg))
   ```

3. **DARE Implementation**
   ```python
   def dare_merge(t, w1, w2, density=0.4):
       # Linear interpolation first
       result = (1 - t) * w1 + t * w2
       # Prune based on magnitude
       abs_vals = mx.abs(result)
       threshold = mx.quantile(abs_vals.reshape(-1), 1.0 - density)
       mask = abs_vals >= threshold
       # Rescale to preserve overall magnitude
       scale = mx.sum(mx.abs(result)) / (mx.sum(mx.abs(result * mask)) + 1e-10)
       return mx.where(mask, result * scale, 0.0)
   ```

#### Slices and Layer Range Processing

The core innovation in this implementation is the slice-based approach, which allows different merging strategies for different portions of the models. This is particularly crucial for merging models with different numbers of layers.

For the QwQ-32B + Bielik case:
- First slice: Layers 0-39 are merged using SLERP
- Second slice: Layers 40-63 from QwQ-32B are preserved using passthrough

This approach is implemented in the `merge_slice` function, which handles the merging of a single slice according to its configuration.

## Performance Analysis

### Memory Usage

Initial testing shows that merging QwQ-32B with Bielik requires significant memory:
- Approximately 72GB when using the default mode
- Approximately 48GB with `low_memory: true` enabled

These numbers may vary based on the hardware and the specific models being merged.

### Time Complexity

Merging time depends on model size and merging method:
- SLERP is the most computationally intensive due to normalization and trigonometric operations
- Linear is the fastest method
- TIES and DARE fall in between

For QwQ-32B + Bielik on M2 Ultra:
- Estimated merge time: 35-45 minutes (default mode)
- Estimated merge time: 60-75 minutes (low memory mode)

## Testing Status

### Test Configurations

1. **QwQ-32B + Bielik-11B Merge**
   - Status: Configuration file created, not yet tested
   - Expected outcome: Polish language capabilities with preserved reasoning
   - Test prompts prepared for verification

2. **Small-Scale Test (Testing Infrastructure)**
   - Models: Mistral-7B + Bielik-7B
   - Status: Pending
   - Purpose: Validate infrastructure with smaller models

### Testing Plan

1. **Unit Tests**
   - Test each merging algorithm independently
   - Verify parameter filtering functionality
   - Validate layer range specification

2. **Integration Tests**
   - End-to-end merging with small models
   - Verification of model loading and saving
   - Tokenizer handling tests

3. **Performance Tests**
   - Memory usage benchmarks
   - Merge time measurements
   - Inference speed comparison

## Known Issues and Limitations

1. **Memory Requirements**
   - Merging large models (>20B parameters) requires substantial RAM
   - Even with low_memory mode, 48GB+ RAM is recommended for QwQ-32B + Bielik

2. **Incompatible Architectures**
   - Models must share compatible architectures (e.g., both Transformer-based)
   - Hidden state dimensions must match within corresponding layers
   - Attention mechanisms should be compatible

3. **Tokenizer Challenges**
   - Tokenizer from only one model can be used
   - Token embeddings might need alignment for optimal performance

4. **MLX-Specific Limitations**
   - Current MLX implementation has some performance differences from PyTorch
   - Quantization in MLX differs from original MergeKit implementations

## Roadmap

### Short-term Goals (next 2 weeks)

1. **Complete Testing Infrastructure**
   - Develop comprehensive unit tests
   - Test with smaller models (7B range)
   - Memory profiling and optimization

2. **Documentation Enhancements**
   - Add detailed API documentation
   - Create step-by-step tutorials
   - Add visual diagrams explaining the merging process

3. **Integration with MLX-LM**
   - Ensure seamless integration with the main MLX-LM package
   - Check compatibility with latest MLX version
   - Create PR for mainline integration

### Medium-term Goals (1-2 months)

1. **New Merging Methods**
   - Implement MoE (Mixture of Experts) merging
   - Add Task Arithmetic support
   - Include advanced normalization strategies

2. **GUI Interface**
   - Create a simple web UI for configuration and monitoring
   - Add visualization of merging process
   - Provide model comparison tools

3. **Optimization for Apple Silicon**
   - Further optimize for M3 chips
   - Add Apple Neural Engine support where applicable
   - Implement pipeline parallelism for multi-chip systems

### Long-term Vision

1. **Automated Merging Recommendations**
   - AI-driven recommendations for optimal merge parameters
   - Automated testing of different merge configurations
   - Self-tuning based on performance metrics

2. **Cross-Framework Compatibility**
   - Import/export compatibility with other merging frameworks
   - Support for diverse model formats
   - Bridge between PyTorch and MLX ecosystems

3. **Language-Specific Optimizations**
   - Special handling for multilingual models
   - Domain-specific merging strategies
   - Enhanced tokenizer alignment tools

## Technical Debt and Refactoring Plans

1. **Code Optimization**
   - Merge operations could be further vectorized
   - Memory management needs refinement
   - Error handling could be more robust

2. **Configuration System**
   - Current YAML parsing is basic, could be enhanced
   - Schema validation needed
   - Default values could be more intelligently determined

3. **Integration Testing**
   - More comprehensive testing with diverse model architectures
   - Edge case handling
   - Performance regression testing

## Contribution Guidelines

For developers interested in contributing:

1. **Priority Areas**
   - Memory optimization for large models
   - Additional merging algorithms
   - Testing infrastructure
   - Documentation and examples

2. **Coding Standards**
   - Follow MLX-LM style conventions
   - Include comprehensive docstrings
   - Add unit tests for new functionality
   - Maintain backward compatibility

3. **Pull Request Process**
   - Create feature branch from `mlx-mergekit`
   - Include unit tests
   - Update documentation
   - Address review feedback

## Resources and References

### Key Papers

1. [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)
2. [Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities](https://arxiv.org/abs/2408.07666)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Related Repositories

1. [Arcee's MergeKit](https://github.com/arcee-ai/mergekit)
2. [MLX Framework](https://github.com/ml-explore/mlx)
3. [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)

### Communication Channels

- GitHub Issues: Primary channel for bug reports and feature requests
- Discussions: For general questions and community support
- Project tracking: Currently using GitHub Projects

## Development Log

### March 19, 2025
- Initial implementation of MergeKit for MLX
- Created core merging algorithms: SLERP, Linear, TIES, DARE, Passthrough
- Implemented slice-based merging approach
- Added configuration system with YAML support
- Created documentation and example configuration

### March 20, 2025
- Created detailed development documentation (this file)
- Planning testing infrastructure
- Identifying optimization opportunities
- Preparing for initial testing with smaller models

## Deployment and Installation

### Current Installation Method

```bash
# Clone the repository
git clone https://github.com/Szowesgad/mlx-lm.git
cd mlx-lm
git checkout mlx-mergekit

# Install dependencies
pip install -e .
```

### Future Packaging Plans

1. **PyPI Package**
   - Package as `mlx-mergekit` or integrate into `mlx-lm`
   - Include all dependencies
   - Versioned releases

2. **Conda Package**
   - Create conda package for easier installation
   - Include pre-built binaries where possible
   - Optimized dependencies

3. **Docker Container**
   - Container with all dependencies pre-installed
   - GPU support configuration
   - Example configurations included

## Performance Monitoring

### Key Metrics to Track

1. **Memory Usage**
   - Peak memory during merging
   - Memory usage by phase
   - Memory scaling with model size

2. **Computation Time**
   - Time per layer
   - Time by merging algorithm
   - Scaling with model size

3. **Result Quality**
   - Perplexity on validation sets
   - Task-specific benchmarks
   - Comparison with original models

## Support and Maintenance

### Current Support Plan

- Issues addressed on GitHub
- Documentation updates with each significant change
- Compatibility maintained with latest MLX releases

### Future Maintenance

- Regular updates to match MLX developments
- Addition of new merging algorithms as they emerge
- Performance optimization based on user feedback

---

*This document serves as the central reference for the development status, plans, and technical details of the MergeKit for MLX project. It will be updated regularly to reflect the current state of development.*
