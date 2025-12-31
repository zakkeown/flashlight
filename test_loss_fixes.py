#!/usr/bin/env python3
"""Test script to verify loss function shape parity fixes."""

import torch
import mlx_compat
import numpy as np


def test_loss_shape(name, torch_fn, mlx_fn, torch_args, mlx_args, **kwargs):
    """Test that a loss function returns the same shape in PyTorch and mlx_compat."""
    # PyTorch
    result_torch = torch_fn(*torch_args, **kwargs)

    # mlx_compat
    result_mlx = mlx_fn(*mlx_args, **kwargs)

    # Compare shapes
    shapes_match = result_torch.shape == tuple(result_mlx.shape)

    status = "✓" if shapes_match else "✗"
    print(f"{status} {name}: PyTorch {result_torch.shape}, mlx_compat {result_mlx.shape}")

    return shapes_match


def main():
    print("Testing loss function shape parity fixes...\n")

    all_passed = True

    # Test 1: binary_cross_entropy_with_logits
    input_t = torch.randn(4, 8)
    target_t = torch.randn(4, 8)
    input_m = mlx_compat.tensor(input_t.numpy())
    target_m = mlx_compat.tensor(target_t.numpy())

    all_passed &= test_loss_shape(
        "binary_cross_entropy_with_logits",
        torch.binary_cross_entropy_with_logits,
        mlx_compat.binary_cross_entropy_with_logits,
        (input_t, target_t),
        (input_m, target_m)
    )

    # Test 2: cosine_embedding_loss
    input1_t = torch.randn(4, 8)
    input2_t = torch.randn(4, 8)
    target_t = torch.ones(4)
    input1_m = mlx_compat.tensor(input1_t.numpy())
    input2_m = mlx_compat.tensor(input2_t.numpy())
    target_m = mlx_compat.tensor(target_t.numpy())

    all_passed &= test_loss_shape(
        "cosine_embedding_loss",
        torch.cosine_embedding_loss,
        mlx_compat.cosine_embedding_loss,
        (input1_t, input2_t, target_t),
        (input1_m, input2_m, target_m)
    )

    # Test 3: hinge_embedding_loss
    input_t = torch.randn(4, 8)
    target_t = torch.ones(4, 8)
    input_m = mlx_compat.tensor(input_t.numpy())
    target_m = mlx_compat.tensor(target_t.numpy())

    all_passed &= test_loss_shape(
        "hinge_embedding_loss",
        torch.hinge_embedding_loss,
        mlx_compat.hinge_embedding_loss,
        (input_t, target_t),
        (input_m, target_m)
    )

    # Test 4: kl_div
    input_t = torch.log(torch.randn(4, 8).abs())
    target_t = torch.randn(4, 8).abs()
    input_m = mlx_compat.tensor(input_t.numpy())
    target_m = mlx_compat.tensor(target_t.numpy())

    all_passed &= test_loss_shape(
        "kl_div",
        torch.kl_div,
        mlx_compat.kl_div,
        (input_t, target_t),
        (input_m, target_m)
    )

    # Test 5: margin_ranking_loss
    input1_t = torch.randn(4, 8)
    input2_t = torch.randn(4, 8)
    target_t = torch.ones(4, 8)
    input1_m = mlx_compat.tensor(input1_t.numpy())
    input2_m = mlx_compat.tensor(input2_t.numpy())
    target_m = mlx_compat.tensor(target_t.numpy())

    all_passed &= test_loss_shape(
        "margin_ranking_loss",
        torch.margin_ranking_loss,
        mlx_compat.margin_ranking_loss,
        (input1_t, input2_t, target_t),
        (input1_m, input2_m, target_m)
    )

    # Test 6: triplet_margin_loss
    anchor_t = torch.randn(4, 8)
    positive_t = torch.randn(4, 8)
    negative_t = torch.randn(4, 8)
    anchor_m = mlx_compat.tensor(anchor_t.numpy())
    positive_m = mlx_compat.tensor(positive_t.numpy())
    negative_m = mlx_compat.tensor(negative_t.numpy())

    all_passed &= test_loss_shape(
        "triplet_margin_loss",
        torch.triplet_margin_loss,
        mlx_compat.triplet_margin_loss,
        (anchor_t, positive_t, negative_t),
        (anchor_m, positive_m, negative_m)
    )

    # Test 7: ctc_loss
    log_probs_t = torch.randn(10, 4, 5).log_softmax(2)
    targets_t = torch.randint(1, 5, (4, 6))
    input_lengths_t = torch.full((4,), 10, dtype=torch.long)
    target_lengths_t = torch.randint(1, 7, (4,), dtype=torch.long)

    log_probs_m = mlx_compat.tensor(log_probs_t.numpy())
    targets_m = mlx_compat.tensor(targets_t.numpy())
    input_lengths_m = mlx_compat.tensor(input_lengths_t.numpy())
    target_lengths_m = mlx_compat.tensor(target_lengths_t.numpy())

    all_passed &= test_loss_shape(
        "ctc_loss",
        torch.ctc_loss,
        mlx_compat.ctc_loss,
        (log_probs_t, targets_t, input_lengths_t, target_lengths_t),
        (log_probs_m, targets_m, input_lengths_m, target_lengths_m)
    )

    print()
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
