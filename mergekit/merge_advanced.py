# Copyright © 2023-2025 Apple Inc.

import argparse
import glob
import shutil
import yaml
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_lm.utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    upload_to_hub,
)

def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Merge multiple models with advanced options.")

    parser.add_argument("--config", type=str, help="Path to the YAML config.")
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_merged_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--low-memory", 
        action="store_true",
        help="Enable memory-saving optimizations."
    )
    return parser

# Original SLERP implementation
def slerp(t, w1, w2, eps=1e-5):
    """
    Spherical linear interpolation

    Args:
        t (float): Interpolation weight in [0.0, 1.0]
        w1 (mx.array): First input
        w2 (mx.array): Second input
        eps (float): Constant for numerical stability
    Returns:
        mx.array: Interpolated result
    """
    t = float(t)
    if t == 0:
        return w1
    elif t == 1:
        return w2
    # Normalize
    v1 = w1 / mx.linalg.norm(w1)
    v2 = w2 / mx.linalg.norm(w2)
    # Angle
    dot = mx.clip((v1 * v2).sum(), 0.0, 1.0)
    theta = mx.arccos(dot)
    sin_theta = mx.sin(theta + eps)
    s1 = mx.sin(theta * (1 - t)) / sin_theta
    s2 = mx.sin(theta * t) / sin_theta
    return s1 * w1 + s2 * w2

# New merge methods
def linear_merge(t, w1, w2):
    """
    Linear interpolation between two weight arrays
    
    Args:
        t (float): Interpolation weight in [0.0, 1.0]
        w1 (mx.array): First input
        w2 (mx.array): Second input
    Returns:
        mx.array: Interpolated result
    """
    return (1 - t) * w1 + t * w2

def ties_merge(t, w1, w2):
    """
    Task Arithmetic with sign-based merging (TIES method)
    
    Args:
        t (float): Interpolation weight in [0.0, 1.0]
        w1 (mx.array): First input
        w2 (mx.array): Second input
    Returns:
        mx.array: Merged result using TIES
    """
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

def dare_merge(t, w1, w2, density=0.4):
    """
    Drop and REscale (DARE) merge method
    
    Args:
        t (float): Interpolation weight in [0.0, 1.0]
        w1 (mx.array): First input
        w2 (mx.array): Second input
        density (float): Target density for pruning (0.0 to 1.0)
    Returns:
        mx.array: Merged result using DARE
    """
    # Linear interpolation first
    result = (1 - t) * w1 + t * w2
    
    # Prune based on magnitude
    abs_vals = mx.abs(result)
    threshold = mx.quantile(abs_vals.reshape(-1), 1.0 - density)
    mask = abs_vals >= threshold
    
    # Rescale to preserve overall magnitude
    scale = mx.sum(mx.abs(result)) / (mx.sum(mx.abs(result * mask)) + 1e-10)
    
    return mx.where(mask, result * scale, 0.0)

def passthrough(w1, w2, source="second"):
    """
    Passthrough one set of weights without merging
    
    Args:
        w1 (mx.array): First input (base model)
        w2 (mx.array): Second input (secondary model)
        source (str): Which weights to pass through ("base" or "second")
    Returns:
        mx.array: Selected weights
    """
    return w1 if source == "base" else w2

def unpack_values(vals, num_layers):
    """
    Unpack values to layer-specific parameters
    
    Args:
        vals: Values to unpack (single value or list of ranges)
        num_layers: Number of layers to cover
    Returns:
        numpy.ndarray: Array of values for each layer
    """
    if isinstance(vals, (int, float)):
        return np.full(num_layers, vals)
    
    bins = len(vals) - 1
    sizes = [num_layers // bins] * bins
    sizes[-1] = num_layers - sum(sizes[:-1])
    
    return np.concatenate(
        [np.linspace(v1, v2, s) for v1, v2, s in zip(vals[:-1], vals[1:], sizes)]
    )

def merge_slice(
    base_model: nn.Module, 
    model: nn.Module, 
    slice_config: dict, 
    verbose: bool = False,
    low_memory: bool = False
):
    """
    Merge a slice of models according to the provided configuration
    
    Args:
        base_model: Base model to merge into
        model: Secondary model to merge from
        slice_config: Configuration for this slice
        verbose: Whether to print verbose information
        low_memory: Whether to optimize for low memory usage
    """
    # Get merge method
    method_name = slice_config.get("merge_method", "slerp")
    if method_name not in ["slerp", "linear", "ties", "dare", "passthrough"]:
        raise ValueError(f"Merge method {method_name} not supported")
    
    # Get layer ranges
    base_range = slice_config.get("sources", [])[0].get("layer_range", None)
    second_range = slice_config.get("sources", [])[1].get("layer_range", None) if len(slice_config.get("sources", [])) > 1 else None
    
    # If no layer range specified, use all layers
    if base_range is None:
        base_range = [0, len(base_model.layers)]
    
    if second_range is None and method_name != "passthrough":
        if len(slice_config.get("sources", [])) > 1:
            second_range = [0, len(model.layers)]
        else:
            # Passthrough with only base model specified
            second_range = base_range
    
    # Check if ranges are valid
    start_base, end_base = base_range
    
    if end_base > len(base_model.layers):
        raise ValueError(f"Base model layer range {base_range} exceeds model dimensions (max: {len(base_model.layers)})")
    
    if method_name != "passthrough" and model is not None:
        start_second, end_second = second_range
        if end_second > len(model.layers):
            raise ValueError(f"Secondary model layer range {second_range} exceeds model dimensions (max: {len(model.layers)})")
        
        # Number of layers to merge
        num_layers = min(end_base - start_base, end_second - start_second)
    else:
        # For passthrough, we only need the base model range
        num_layers = end_base - start_base
    
    # Get parameters
    if method_name != "passthrough":
        param_list = slice_config.get("parameters", {}).get("t", [])
        params = {}
        filter_keys = set()
        
        # Default is the last item without a filter
        default_value = 0.5
        
        if param_list:
            # Process all parameter definitions except the last one
            for pl in param_list[:-1]:
                if "filter" in pl:
                    params[pl["filter"]] = unpack_values(pl["value"], num_layers)
                    filter_keys.add(pl["filter"])
            
            # Get default from the last parameter if it exists
            if param_list and "value" in param_list[-1]:
                default_value = param_list[-1]["value"]
                
        default = unpack_values(default_value, num_layers)
    
    # Perform merging based on the method
    if verbose:
        print(f"Merging slice using method: {method_name}")
        print(f"  Base model layers: {base_range}")
        if method_name != "passthrough" and model is not None:
            print(f"  Secondary model layers: {second_range}")
    
    # For each layer in the range
    for i in range(num_layers):
        base_idx = start_base + i
        
        if method_name == "passthrough":
            if verbose:
                print(f"  Passthrough layer {base_idx}")
            
            # For passthrough, check which source to use
            source = slice_config.get("sources", [])[0].get("model", "base")
            
            # If source is not base, we need to copy the weights from the secondary model
            if source != "base" and len(slice_config.get("sources", [])) > 1 and model is not None:
                second_idx = start_second + i
                bl = base_model.layers[base_idx]
                l = model.layers[second_idx]
                base_weights = bl.parameters()
                weights = l.parameters()
                
                # In low memory mode, process one parameter at a time
                if low_memory:
                    for k, w1 in base_weights.items():
                        w2 = weights[k]
                        base_weights[k] = passthrough(w1, w2, source="second")
                        base_model.update({f"layers.{base_idx}.{k}": base_weights[k]})
                        # Free memory
                        base_weights[k] = None
                else:
                    for k, w1 in base_weights.items():
                        w2 = weights[k]
                        base_weights[k] = passthrough(w1, w2, source="second")
                    
                    base_model.update({f"layers.{base_idx}.{k}": v for k, v in weights.items()})
            # Otherwise, we keep the base model weights (do nothing)
        else:
            second_idx = start_second + i
            bl = base_model.layers[base_idx]
            l = model.layers[second_idx]
            base_weights = bl.parameters()
            weights = l.parameters()
            
            # In low memory mode, process one parameter at a time
            if low_memory:
                for k, w1 in base_weights.items():
                    w2 = weights[k]
                    
                    # Determine interpolation parameter t
                    t = params.get(k, default)[i] if any(fk in k for fk in filter_keys) else default[i]
                    
                    # Apply merge method
                    if method_name == "slerp":
                        result = slerp(t, w1, w2)
                    elif method_name == "linear":
                        result = linear_merge(t, w1, w2)
                    elif method_name == "ties":
                        result = ties_merge(t, w1, w2)
                    elif method_name == "dare":
                        density = slice_config.get("parameters", {}).get("density", 0.4)
                        result = dare_merge(t, w1, w2, density)
                    
                    # Update and free memory
                    base_model.update({f"layers.{base_idx}.{k}": result})
                    result = None
            else:
                for k, w1 in base_weights.items():
                    w2 = weights[k]
                    
                    # Determine interpolation parameter t
                    t = params.get(k, default)[i] if any(fk in k for fk in filter_keys) else default[i]
                    
                    # Apply merge method
                    if method_name == "slerp":
                        result = slerp(t, w1, w2)
                    elif method_name == "linear":
                        result = linear_merge(t, w1, w2)
                    elif method_name == "ties":
                        result = ties_merge(t, w1, w2)
                    elif method_name == "dare":
                        density = slice_config.get("parameters", {}).get("density", 0.4)
                        result = dare_merge(t, w1, w2, density)
                    
                    base_weights[k] = result
                
                base_model.update(base_weights)

def merge_advanced(
    config: str,
    mlx_path: str = "mlx_model",
    upload_repo: Optional[str] = None,
    verbose: bool = False,
    low_memory: bool = False,
):
    """
    Merge multiple models with advanced options
    
    Args:
        config: Path to YAML configuration file
        mlx_path: Path to save merged model
        upload_repo: Optional Hugging Face repo to upload the model to
        verbose: Whether to print verbose information
        low_memory: Whether to optimize for low memory usage
    """
    with open(config, "r") as fid:
        merge_conf = yaml.safe_load(fid)
    
    if verbose:
        print("[INFO] Starting advanced model merge")
    
    # Get advanced options
    advanced_options = merge_conf.get("advanced", {})
    if advanced_options.get("low_memory", False):
        low_memory = True
    
    # Load model definitions
    models_config = merge_conf.get("models", [])
    if len(models_config) < 1:
        raise ValueError(f"Expected at least 1 model, got {len(models_config)}.")
    
    # Determine base model
    base_model_idx = 0
    for i, model_config in enumerate(models_config):
        if model_config.get("base_model", False):
            base_model_idx = i
            break
    
    # Load base model
    base_model_config = models_config[base_model_idx]
    base_hf_path = base_model_config.get("path", "")
    base_path = get_model_path(base_hf_path)
    
    if verbose:
        print(f"[INFO] Loading base model from {base_hf_path}")
    
    base_model, base_config, tokenizer = fetch_from_hub(base_path, lazy=True)
    
    # Load additional models
    models = []
    configs = []
    tokenizers = []
    model_names = {}
    
    for i, model_config in enumerate(models_config):
        model_name = model_config.get("name", f"model_{i}")
        model_names[model_name] = i
        
        if i == base_model_idx:
            # Skip base model as it's already loaded
            models.append(None)
            configs.append(None)
            tokenizers.append(None)
            continue
        
        model_path = model_config.get("path", "")
        
        if verbose:
            print(f"[INFO] Loading additional model from {model_path}")
        
        model, model_config, model_tokenizer = fetch_from_hub(get_model_path(model_path), lazy=True)
        models.append(model)
        configs.append(model_config)
        tokenizers.append(model_tokenizer)
    
    # Process slices
    slices = merge_conf.get("slices", [])
    if not slices:
        # If no slices defined, create a default slice for backward compatibility
        slices = [{
            "sources": [
                {"model": base_model_config.get("name", "base"), "layer_range": [0, len(base_model.layers)]},
            ],
            "merge_method": merge_conf.get("method", "slerp"),
            "parameters": merge_conf.get("parameters", {}),
        }]
        
        # Add second model if available
        if len(models) > 0 and models[0] is not None:
            second_model_idx = 0 if base_model_idx != 0 else 1
            second_model_name = models_config[second_model_idx].get("name", f"model_{second_model_idx}")
            second_model = models[0]
            
            slices[0]["sources"].append({
                "model": second_model_name,
                "layer_range": [0, len(second_model.layers)]
            })
    
    # Process each slice
    for i, slice_config in enumerate(slices):
        if verbose:
            print(f"[INFO] Processing slice {i+1}/{len(slices)}")
        
        # Get model indices for this slice
        sources = slice_config.get("sources", [])
        
        if not sources:
            raise ValueError(f"Slice {i} has no sources defined")
        
        # First source is always used with the base model
        first_source_model_name = sources[0].get("model", "")
        
        # For the second source, find the model by name
        if len(sources) > 1:
            second_model_name = sources[1].get("model", "")
            
            # Find the index of the second model
            second_model_idx = model_names.get(second_model_name)
            
            if second_model_idx is None:
                raise ValueError(f"Model {second_model_name} not found")
            
            # Get the second model
            if second_model_idx == base_model_idx:
                # This is a special case where we're merging the base model with itself
                # Useful for certain types of self-distillation or adaptation
                merge_slice(base_model, base_model, slice_config, verbose, low_memory)
            else:
                # Adjust index to account for the None entry in models list
                adj_idx = second_model_idx if second_model_idx < base_model_idx else second_model_idx - 1
                merge_slice(base_model, models[adj_idx], slice_config, verbose, low_memory)
        else:
            # Only one source, must be passthrough
            slice_config["merge_method"] = "passthrough"
            merge_slice(base_model, None, slice_config, verbose, low_memory)
    
    # Handle tokenizer
    tokenizer_config = merge_conf.get("tokenizer", {})
    tokenizer_source = tokenizer_config.get("source", "")
    
    # If tokenizer source is specified and not the base model
    if tokenizer_source and tokenizer_source != base_model_config.get("name", "base"):
        source_idx = model_names.get(tokenizer_source)
        if source_idx is not None and source_idx != base_model_idx:
            # Adjust index to account for the None entry in tokenizers list
            adj_idx = source_idx if source_idx < base_model_idx else source_idx - 1
            if tokenizers[adj_idx] is not None:
                tokenizer = tokenizers[adj_idx]
                if verbose:
                    print(f"[INFO] Using tokenizer from {tokenizer_source}")
    
    # Save the merged model
    if verbose:
        print(f"[INFO] Saving merged model to {mlx_path}")
    
    mlx_path = Path(mlx_path)
    weights = dict(tree_flatten(base_model.parameters()))
    
    # Clean up memory
    del models, base_model
    
    save_weights(mlx_path, weights, donate_weights=True)
    
    # Copy Python files
    py_files = glob.glob(str(base_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(mlx_path)
    
    # Save config
    save_config(config, config_path=mlx_path / "config.json")
    
    # Copy original merge config for reference
    with open(mlx_path / "merge_config.yaml", "w") as f:
        yaml.dump(merge_conf, f)
    
    if upload_repo is not None:
        if verbose:
            print(f"[INFO] Uploading model to {upload_repo}")
        
        upload_to_hub(mlx_path, upload_repo, base_hf_path)

def main():
    parser = configure_parser()
    args = parser.parse_args()
    merge_advanced(**vars(args))

if __name__ == "__main__":
    main()
