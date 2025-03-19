#!/usr/bin/env python3
# Copyright © 2023-2025 Apple Inc.

"""
Test script for MergeKit functionality
"""
import argparse
import os
import time
import tempfile
import yaml
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mergekit import mergekit

def parse_args():
    parser = argparse.ArgumentParser(description="Test MergeKit functionality")
    parser.add_argument("--config", type=str, required=True, help="Path to merge config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for merged model")
    parser.add_argument("--test-prompt", type=str, default="Explain quantum computing in simple terms.", 
                        help="Test prompt to run through the merged model")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step and only test existing model")
    parser.add_argument("--no-test", action="store_true", help="Skip testing the merged model")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    return parser.parse_args()

def load_and_test_model(model_path, test_prompt, max_tokens, verbose=False):
    """Load a model and test it with a prompt"""
    try:
        # Import generate from mlx_lm
        from mlx_lm.generate import generate, load
        
        if verbose:
            print(f"Loading model from {model_path}")
        
        start_time = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - start_time
        
        if verbose:
            print(f"Model loaded in {load_time:.2f} seconds")
            print(f"Testing model with prompt: '{test_prompt}'")
        
        # Generate text with the model
        generate_start = time.time()
        
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_tokens=max_tokens,
            temp=0.7,
            verbose=verbose,
        )
        
        generate_time = time.time() - generate_start
        
        if verbose:
            print(f"Generation completed in {generate_time:.2f} seconds")
            print(f"Generated {len(result)} tokens at {len(result)/generate_time:.2f} tokens/sec")
        
        return True, load_time, generate_time, len(result)
    except Exception as e:
        print(f"Error testing model: {e}")
        return False, 0, 0, 0

def run_benchmark(model_path, verbose=False):
    """Run performance benchmarks on the merged model"""
    try:
        # Import necessary functions
        from mlx_lm.generate import load
        import psutil
        
        # Load the model and measure memory usage
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        if verbose:
            print(f"Memory usage before loading: {mem_before:.2f} MB")
            print(f"Loading model from {model_path}")
        
        start_time = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - start_time
        
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_delta = mem_after - mem_before
        
        if verbose:
            print(f"Model loaded in {load_time:.2f} seconds")
            print(f"Memory usage after loading: {mem_after:.2f} MB")
            print(f"Memory delta: {mem_delta:.2f} MB")
        
        # Test inference speed with different sequence lengths
        seq_lengths = [100, 200, 500, 1000]
        results = {}
        
        for seq_len in seq_lengths:
            if verbose:
                print(f"Testing inference with {seq_len} tokens...")
            
            # Create a dummy prompt that will force generation of seq_len tokens
            prompt = "Continue the story: Once upon a time"
            
            # Tokenize the prompt
            input_ids = tokenizer.encode(prompt)
            input_len = len(input_ids)
            
            # Measure inference time
            start_time = time.time()
            
            # Run generation manually to measure per-token generation time
            for _ in range(min(seq_len, 25)):  # Limit to 25 tokens for benchmarking
                # Simple forward pass
                logits = model(mx.array([input_ids]))
                
                # Get the last token
                new_token = logits[0, -1].argmax().item()
                input_ids.append(new_token)
            
            inference_time = time.time() - start_time
            tokens_per_sec = min(seq_len, 25) / inference_time
            
            results[seq_len] = {
                "tokens_per_sec": tokens_per_sec,
                "inference_time": inference_time
            }
            
            if verbose:
                print(f"  Generated {min(seq_len, 25)} tokens in {inference_time:.2f} seconds")
                print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
        
        benchmark_results = {
            "memory": {
                "before_load_mb": mem_before,
                "after_load_mb": mem_after,
                "delta_mb": mem_delta
            },
            "load_time_seconds": load_time,
            "inference": results
        }
        
        # Save benchmark results
        benchmark_path = Path(model_path) / "benchmark_results.yaml"
        with open(benchmark_path, "w") as f:
            yaml.dump(benchmark_results, f, default_flow_style=False)
        
        if verbose:
            print(f"Benchmark results saved to {benchmark_path}")
        
        return benchmark_results
    
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        # Create temporary directory if none specified
        temp_dir = tempfile.mkdtemp(prefix="mlx_merged_")
        output_dir = temp_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    if not args.skip_merge:
        print(f"Starting model merge using config: {args.config}")
        start_time = time.time()
        
        # Run merge
        mergekit(
            config=args.config,
            mlx_path=str(output_dir),
            verbose=args.verbose
        )
        
        merge_time = time.time() - start_time
        print(f"Merge completed in {merge_time:.2f} seconds")
    else:
        print(f"Skipping merge, using existing model at: {output_dir}")
    
    # Test the merged model if requested
    if not args.no_test:
        success, load_time, generate_time, num_tokens = load_and_test_model(
            model_path=str(output_dir),
            test_prompt=args.test_prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose
        )
        
        if success:
            print("\nModel test successful!")
            print(f"Load time: {load_time:.2f} seconds")
            print(f"Generation time: {generate_time:.2f} seconds")
            print(f"Tokens generated: {num_tokens}")
            print(f"Tokens per second: {num_tokens/generate_time:.2f}")
        else:
            print("\nModel test failed!")
    
    # Run benchmarks if requested
    if args.benchmark:
        print("\nRunning performance benchmarks...")
        benchmark_results = run_benchmark(
            model_path=str(output_dir),
            verbose=args.verbose
        )
        
        if benchmark_results:
            print("\nBenchmark summary:")
            print(f"Memory usage for model: {benchmark_results['memory']['delta_mb']:.2f} MB")
            print(f"Model load time: {benchmark_results['load_time_seconds']:.2f} seconds")
            
            # Print inference results
            print("\nInference speed:")
            for seq_len, results in benchmark_results['inference'].items():
                print(f"  {seq_len} tokens: {results['tokens_per_sec']:.2f} tokens/sec")
    
    print(f"\nModel available at: {output_dir}")

if __name__ == "__main__":
    main()
