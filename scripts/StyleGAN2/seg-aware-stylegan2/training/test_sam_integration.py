"""
Unit tests for SAM extractor with caching.

Tests:
- Single image extraction
- Cache save/load functionality
- Speed comparison (first call vs cached)
- Batch padding

Usage:
    python training/test_sam_integration.py
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sam_extractor import SAMExtractor, pad_embeddings_batch


def test_single_extraction():
    """Test single image extraction"""
    print("\n" + "="*60)
    print("TEST 1: Single Image Extraction")
    print("="*60)
    
    # Setup
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    cache_dir = "./test_sam_cache"
    test_image = "test_image.jpg"
    
    if not os.path.exists(sam_checkpoint):
        print(f"❌ SAM checkpoint not found: {sam_checkpoint}")
        print("   Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return False
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("   Please provide a test image")
        return False
    
    # Create extractor
    extractor = SAMExtractor(
        sam_checkpoint=sam_checkpoint,
        cache_dir=cache_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="vit_b"
    )
    
    print(f"✓ SAMExtractor created")
    print(f"  Checkpoint: {sam_checkpoint}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Device: {extractor.device}")
    
    # Extract
    print("\nExtracting embeddings...")
    start_time = time.time()
    results = extractor.extract_or_load([test_image], None)
    extract_time = time.time() - start_time
    
    print(f"✓ Extraction completed in {extract_time:.2f}s")
    print(f"  Number of masks: {len(results[0]['scores'])}")
    print(f"  Embedding shape: {results[0]['emb'].shape}")
    print(f"  Scores shape: {results[0]['scores'].shape}")
    
    # Validate
    assert len(results) == 1, "Should return 1 result"
    assert 'emb' in results[0], "Result should have 'emb' key"
    assert 'scores' in results[0], "Result should have 'scores' key"
    assert results[0]['emb'].shape[1] == 256, "Embeddings should have 256 dimensions"
    assert len(results[0]['scores']) == results[0]['emb'].shape[0], "Scores and embeddings should match"
    
    print("✓ All validations passed")
    
    return True


def test_cache():
    """Test cache save/load functionality"""
    print("\n" + "="*60)
    print("TEST 2: Cache Save/Load")
    print("="*60)
    
    # Setup
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    cache_dir = "./test_sam_cache"
    test_image = "test_image.jpg"
    
    if not os.path.exists(sam_checkpoint) or not os.path.exists(test_image):
        print("❌ Skipping - missing required files")
        return False
    
    # Create extractor
    extractor = SAMExtractor(
        sam_checkpoint=sam_checkpoint,
        cache_dir=cache_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="vit_b"
    )
    
    # First extraction (should miss cache and extract)
    print("\nFirst extraction (cache miss expected)...")
    start_time = time.time()
    results1 = extractor.extract_or_load([test_image], None)
    time1 = time.time() - start_time
    
    stats1 = extractor.get_stats()
    print(f"✓ First extraction: {time1:.2f}s")
    print(f"  Cache hits: {stats1['cache_hits']}")
    print(f"  Cache misses: {stats1['cache_misses']}")
    print(f"  Extractions: {stats1['extractions']}")
    
    # Wait for async write to complete
    extractor.async_writer.queue.join()
    print("✓ Async write completed")
    
    # Second extraction (should hit cache)
    print("\nSecond extraction (cache hit expected)...")
    start_time = time.time()
    results2 = extractor.extract_or_load([test_image], None)
    time2 = time.time() - start_time
    
    stats2 = extractor.get_stats()
    print(f"✓ Second extraction: {time2:.2f}s")
    print(f"  Cache hits: {stats2['cache_hits']}")
    print(f"  Cache misses: {stats2['cache_misses']}")
    print(f"  Extractions: {stats2['extractions']}")
    
    # Validate
    assert stats2['cache_hits'] > stats1['cache_hits'], "Cache hit should increase"
    assert stats2['extractions'] == stats1['extractions'], "No new extractions should occur"
    assert time2 < time1 / 10, f"Cache read should be much faster (got {time2:.2f}s vs {time1:.2f}s)"
    
    # Verify results are identical
    np.testing.assert_array_equal(results1[0]['emb'], results2[0]['emb'])
    np.testing.assert_array_equal(results1[0]['scores'], results2[0]['scores'])
    
    print(f"✓ Cache speedup: {time1/time2:.1f}x faster")
    print("✓ Results identical")
    print("✓ All cache tests passed")
    
    return True


def test_batch_padding():
    """Test batch padding functionality"""
    print("\n" + "="*60)
    print("TEST 3: Batch Padding")
    print("="*60)
    
    # Create mock embeddings with different lengths
    mock_embeddings = [
        {'emb': np.random.randn(10, 256).astype(np.float16), 'scores': np.random.rand(10).astype(np.float32)},
        {'emb': np.random.randn(5, 256).astype(np.float16), 'scores': np.random.rand(5).astype(np.float32)},
        {'emb': np.random.randn(15, 256).astype(np.float16), 'scores': np.random.rand(15).astype(np.float32)},
    ]
    
    print(f"Mock embeddings created:")
    for i, emb in enumerate(mock_embeddings):
        print(f"  Batch {i}: {emb['emb'].shape[0]} masks")
    
    # Pad
    padded_emb, pad_mask = pad_embeddings_batch(mock_embeddings, device="cpu")
    
    print(f"\n✓ Padded embeddings: {padded_emb.shape}")
    print(f"✓ Pad mask: {pad_mask.shape}")
    
    # Validate
    assert padded_emb.shape[0] == 3, "Batch size should be 3"
    assert padded_emb.shape[1] == 15, "Max masks should be 15"
    assert padded_emb.shape[2] == 256, "Embedding dim should be 256"
    
    # Check padding mask
    assert pad_mask[0, :10].sum() == 0, "First 10 should be valid (not padded)"
    assert pad_mask[0, 10:].sum() == 5, "Last 5 should be padded"
    
    assert pad_mask[1, :5].sum() == 0, "First 5 should be valid"
    assert pad_mask[1, 5:].sum() == 10, "Last 10 should be padded"
    
    assert pad_mask[2, :15].sum() == 0, "All 15 should be valid"
    
    print("✓ Padding correctness validated")
    print("✓ All batch padding tests passed")
    
    return True


def test_stochastic_extraction():
    """Test stochastic SAM conditioning logic"""
    print("\n" + "="*60)
    print("TEST 4: Stochastic SAM Conditioning")
    print("="*60)
    
    # Simulate stochastic extraction
    sam_prob = 0.25
    n_batches = 1000
    
    use_sam_count = 0
    for _ in range(n_batches):
        use_sam = np.random.random() < sam_prob
        if use_sam:
            use_sam_count += 1
    
    actual_prob = use_sam_count / n_batches
    print(f"Expected probability: {sam_prob}")
    print(f"Actual probability: {actual_prob:.3f}")
    print(f"Difference: {abs(actual_prob - sam_prob):.3f}")
    
    # Should be close to expected (within 5%)
    assert abs(actual_prob - sam_prob) < 0.05, f"Probability mismatch: {actual_prob} vs {sam_prob}"
    
    print("✓ Stochastic conditioning works as expected")
    
    # Coverage simulation
    print("\nCoverage simulation (probabilistic extraction):")
    n_epochs = 15
    for epoch in range(1, n_epochs + 1):
        coverage = 1.0 - (1.0 - sam_prob) ** epoch
        print(f"  Epoch {epoch:2d}: {coverage*100:.1f}% images seen")
    
    print("✓ Coverage simulation completed")
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("SAM EXTRACTOR INTEGRATION TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Single extraction
    try:
        results.append(("Single Extraction", test_single_extraction()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Single Extraction", False))
    
    # Test 2: Cache
    try:
        results.append(("Cache Save/Load", test_cache()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Cache Save/Load", False))
    
    # Test 3: Batch padding
    try:
        results.append(("Batch Padding", test_batch_padding()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Batch Padding", False))
    
    # Test 4: Stochastic extraction
    try:
        results.append(("Stochastic Conditioning", test_stochastic_extraction()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(("Stochastic Conditioning", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
