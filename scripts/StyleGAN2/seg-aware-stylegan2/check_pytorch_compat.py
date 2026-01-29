#!/usr/bin/env python3
"""
PyTorch 2.7.0 Compatibility Check Script
Verifies that all critical operations work correctly with PyTorch 2.x
"""

import torch
import sys

def check_version():
    """Check PyTorch version"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print()

def check_jit_operations():
    """Check if JIT operations are accessible"""
    print("=" * 60)
    print("Checking JIT operations...")
    print("=" * 60)
    
    operations = [
        'aten::grid_sampler_2d_backward',
        'aten::cudnn_convolution_backward',
        'aten::cudnn_convolution_backward_weight',
        'aten::cudnn_convolution_transpose_backward_weight'
    ]
    
    for op_name in operations:
        try:
            op = torch._C._jit_get_operation(op_name)
            is_tuple = isinstance(op, tuple)
            print(f"✓ {op_name}: {'tuple' if is_tuple else 'callable'}")
            if is_tuple and len(op) > 1:
                print(f"  Schema: {str(op[1])[:100]}...")
        except Exception as e:
            print(f"✗ {op_name}: {e}")
    print()

def check_grid_sample():
    """Test grid_sample backward operation"""
    print("=" * 60)
    print("Testing grid_sample backward...")
    print("=" * 60)
    
    try:
        # Create test tensors
        batch_size = 2
        channels = 3
        h, w = 8, 8
        
        input_tensor = torch.randn(batch_size, channels, h, w, requires_grad=True)
        grid = torch.randn(batch_size, h, w, 2)
        
        # Forward pass
        output = torch.nn.functional.grid_sample(
            input_tensor, grid, mode='bilinear', 
            padding_mode='zeros', align_corners=False
        )
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        print("✓ grid_sample forward/backward works")
        
        # Test with JIT operation directly
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        if isinstance(op, tuple):
            op = op[0]
        
        # Try PyTorch 2.x signature with output_mask
        try:
            grad_input, grad_grid = op(grad_output, input_tensor.detach(), grid, 0, 0, False, [True, True])
            print("✓ grid_sampler_2d_backward with output_mask works (PyTorch 2.x)")
        except TypeError as e:
            # Try PyTorch 1.x signature
            try:
                grad_input, grad_grid = op(grad_output, input_tensor.detach(), grid, 0, 0, False)
                print("✓ grid_sampler_2d_backward without output_mask works (PyTorch 1.x)")
            except Exception as e2:
                print(f"✗ grid_sampler_2d_backward failed: {e2}")
                return False
        
    except Exception as e:
        print(f"✗ grid_sample test failed: {e}")
        return False
    
    print()
    return True

def check_conv2d():
    """Test conv2d backward operations"""
    print("=" * 60)
    print("Testing conv2d backward...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping cudnn tests")
        print()
        return True
    
    try:
        device = torch.device('cuda')
        
        # Create test tensors
        batch_size = 2
        in_channels = 3
        out_channels = 16
        h, w = 8, 8
        kernel_size = 3
        
        input_tensor = torch.randn(batch_size, in_channels, h, w, device=device, requires_grad=True)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device, requires_grad=True)
        
        # Forward pass
        output = torch.nn.functional.conv2d(input_tensor, weight, stride=1, padding=1)
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        print("✓ conv2d forward/backward works")
        print()
        
    except Exception as e:
        print(f"✗ conv2d test failed: {e}")
        print()
        return False
    
    return True

def check_custom_ops():
    """Check custom ops compatibility"""
    print("=" * 60)
    print("Checking custom ops imports...")
    print("=" * 60)
    
    try:
        from torch_utils.ops import conv2d_gradfix
        print(f"✓ conv2d_gradfix imported (enabled={conv2d_gradfix.enabled})")
    except Exception as e:
        print(f"✗ conv2d_gradfix import failed: {e}")
    
    try:
        from torch_utils.ops import grid_sample_gradfix
        print(f"✓ grid_sample_gradfix imported (enabled={grid_sample_gradfix.enabled})")
    except Exception as e:
        print(f"✗ grid_sample_gradfix import failed: {e}")
    
    try:
        from torch_utils.ops import upfirdn2d
        print("✓ upfirdn2d imported")
    except Exception as e:
        print(f"✗ upfirdn2d import failed: {e}")
    
    try:
        from torch_utils.ops import bias_act
        print("✓ bias_act imported")
    except Exception as e:
        print(f"✗ bias_act import failed: {e}")
    
    print()

def main():
    print("\n" + "=" * 60)
    print("PyTorch 2.7.0 Compatibility Check")
    print("=" * 60 + "\n")
    
    check_version()
    check_jit_operations()
    
    grid_sample_ok = check_grid_sample()
    conv2d_ok = check_conv2d()
    check_custom_ops()
    
    print("=" * 60)
    if grid_sample_ok and conv2d_ok:
        print("✓ All critical tests passed!")
    else:
        print("✗ Some tests failed - check output above")
    print("=" * 60)

if __name__ == "__main__":
    main()
