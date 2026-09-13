"""
scripts/download_weights.py
===========================
Downloads the pre-trained Mini-Xception FER2013 weights and converts
them from Keras .hdf5 format to a PyTorch state_dict (.pt) file.

Usage
-----
    python scripts/download_weights.py

The script will:
  1. Download the original Keras weights from the face_classification
     GitHub release (~1.2 MB).
  2. Convert weight tensors to PyTorch format (layer-by-layer mapping).
  3. Save to weights/fer_mini_xception.pt in the project root.

Requirements
------------
  pip install torch h5py requests
  (tensorflow / keras NOT required — we read .hdf5 directly via h5py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make sure project root is importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUT_PATH = WEIGHTS_DIR / "fer_mini_xception.pt"

# Original Keras HDF5 weights from oarriaga/face_classification GitHub release
HDF5_URL = (
    "https://github.com/oarriaga/face_classification/releases/download/"
    "v1.0/fer2013_mini_XCEPTION.102-0.66.hdf5"
)
HDF5_CACHE = WEIGHTS_DIR / "fer2013_mini_XCEPTION.102-0.66.hdf5"


def download_hdf5(url: str, dest: Path) -> None:
    """Download a file from url to dest with a progress bar."""
    import requests

    print(f"Downloading weights from:\n  {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8192

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded // 1024} KB / {total // 1024} KB  ({pct:.1f}%)", end="")
    print(f"\nSaved to: {dest}")


def convert_hdf5_to_pytorch(hdf5_path: Path, output_path: Path) -> None:
    """
    Convert Mini-Xception Keras HDF5 weights to a PyTorch state_dict.

    The Keras model from oarriaga/face_classification uses:
      - Block naming: conv2d, batch_normalization, separable_conv2d, dense
    We map these layer-by-layer to our MiniXception module hierarchy.
    """
    try:
        import h5py
        import torch
        import numpy as np
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install h5py torch")
        sys.exit(1)

    from visionedit.streams.fer_model import MiniXception

    print(f"\nConverting {hdf5_path.name} → PyTorch state_dict ...")

    model = MiniXception()
    state = model.state_dict()

    with h5py.File(hdf5_path, "r") as f:
        # Print available layer groups to help debugging if mapping fails
        if "model_weights" in f:
            weight_group = f["model_weights"]
        else:
            weight_group = f

        def _get_weights(layer_name: str, weight_name: str) -> np.ndarray:
            """Extract a weight array by Keras layer/weight name."""
            try:
                # Keras h5 structure: model_weights/<layer>/<layer>/<weight>
                grp = weight_group[layer_name][layer_name]
                return np.array(grp[weight_name])
            except KeyError:
                try:
                    grp = weight_group[layer_name]
                    return np.array(grp[weight_name])
                except KeyError:
                    raise KeyError(
                        f"Could not find '{layer_name}/{weight_name}' in HDF5. "
                        "The HDF5 structure may differ from expected. "
                        "Run with --debug to inspect the HDF5 file structure."
                    )

        def _keras_to_pt_conv(w: np.ndarray) -> np.ndarray:
            """Keras conv weight: (kH, kW, Cin, Cout) → PyTorch: (Cout, Cin, kH, kW)."""
            return np.transpose(w, (3, 2, 0, 1))

        def _keras_to_pt_dw(w: np.ndarray) -> np.ndarray:
            """Keras depthwise conv weight: (kH, kW, Cin, depth_mult) → (Cin*depth_mult, 1, kH, kW)."""
            # For depth_multiplier=1: (kH, kW, Cin, 1) → (Cin, 1, kH, kW)
            w = np.transpose(w, (2, 3, 0, 1))
            return w

        try:
            # ── Entry block ───────────────────────────────────────────────────
            state["entry.0.weight"] = torch.tensor(
                _keras_to_pt_conv(_get_weights("conv2d", "kernel:0"))
            )
            state["entry.1.weight"] = torch.tensor(_get_weights("batch_normalization", "gamma:0"))
            state["entry.1.bias"]   = torch.tensor(_get_weights("batch_normalization", "beta:0"))
            state["entry.1.running_mean"] = torch.tensor(_get_weights("batch_normalization", "moving_mean:0"))
            state["entry.1.running_var"]  = torch.tensor(_get_weights("batch_normalization", "moving_variance:0"))

            state["entry.3.weight"] = torch.tensor(
                _keras_to_pt_conv(_get_weights("conv2d_1", "kernel:0"))
            )
            state["entry.4.weight"] = torch.tensor(_get_weights("batch_normalization_1", "gamma:0"))
            state["entry.4.bias"]   = torch.tensor(_get_weights("batch_normalization_1", "beta:0"))
            state["entry.4.running_mean"] = torch.tensor(_get_weights("batch_normalization_1", "moving_mean:0"))
            state["entry.4.running_var"]  = torch.tensor(_get_weights("batch_normalization_1", "moving_variance:0"))

            print("  ✓ Entry block converted")

            # ── Residual blocks 1–4 ───────────────────────────────────────────
            block_map = [
                # (block_attr, dsc_prefix, shortcut_conv_prefix, shortcut_bn_prefix)
                ("block1", "separable_conv2d",   "separable_conv2d_1",   "conv2d_2",   "batch_normalization_2",
                           "batch_normalization_3", "batch_normalization_4"),
                ("block2", "separable_conv2d_2", "separable_conv2d_3",   "conv2d_3",   "batch_normalization_5",
                           "batch_normalization_6", "batch_normalization_7"),
                ("block3", "separable_conv2d_4", "separable_conv2d_5",   "conv2d_4",   "batch_normalization_8",
                           "batch_normalization_9", "batch_normalization_10"),
                ("block4", "separable_conv2d_6", "separable_conv2d_7",   "conv2d_5",   "batch_normalization_11",
                           "batch_normalization_12", "batch_normalization_13"),
            ]

            for block_attr, dsc1_name, dsc2_name, sc_conv_name, dsc1_bn_name, dsc2_bn_name, sc_bn_name in block_map:
                # dsc1 depthwise
                state[f"{block_attr}.dsc1.depthwise.weight"] = torch.tensor(
                    _keras_to_pt_dw(_get_weights(dsc1_name, "depthwise_kernel:0"))
                )
                state[f"{block_attr}.dsc1.pointwise.weight"] = torch.tensor(
                    _keras_to_pt_conv(_get_weights(dsc1_name, "pointwise_kernel:0"))
                )
                state[f"{block_attr}.dsc1.bn.weight"] = torch.tensor(_get_weights(dsc1_bn_name, "gamma:0"))
                state[f"{block_attr}.dsc1.bn.bias"]   = torch.tensor(_get_weights(dsc1_bn_name, "beta:0"))
                state[f"{block_attr}.dsc1.bn.running_mean"] = torch.tensor(_get_weights(dsc1_bn_name, "moving_mean:0"))
                state[f"{block_attr}.dsc1.bn.running_var"]  = torch.tensor(_get_weights(dsc1_bn_name, "moving_variance:0"))

                # dsc2 depthwise
                state[f"{block_attr}.dsc2.depthwise.weight"] = torch.tensor(
                    _keras_to_pt_dw(_get_weights(dsc2_name, "depthwise_kernel:0"))
                )
                state[f"{block_attr}.dsc2.pointwise.weight"] = torch.tensor(
                    _keras_to_pt_conv(_get_weights(dsc2_name, "pointwise_kernel:0"))
                )
                state[f"{block_attr}.dsc2.bn.weight"] = torch.tensor(_get_weights(dsc2_bn_name, "gamma:0"))
                state[f"{block_attr}.dsc2.bn.bias"]   = torch.tensor(_get_weights(dsc2_bn_name, "beta:0"))
                state[f"{block_attr}.dsc2.bn.running_mean"] = torch.tensor(_get_weights(dsc2_bn_name, "moving_mean:0"))
                state[f"{block_attr}.dsc2.bn.running_var"]  = torch.tensor(_get_weights(dsc2_bn_name, "moving_variance:0"))

                # Shortcut conv
                state[f"{block_attr}.shortcut.0.weight"] = torch.tensor(
                    _keras_to_pt_conv(_get_weights(sc_conv_name, "kernel:0"))
                )
                state[f"{block_attr}.shortcut.1.weight"] = torch.tensor(_get_weights(sc_bn_name, "gamma:0"))
                state[f"{block_attr}.shortcut.1.bias"]   = torch.tensor(_get_weights(sc_bn_name, "beta:0"))
                state[f"{block_attr}.shortcut.1.running_mean"] = torch.tensor(_get_weights(sc_bn_name, "moving_mean:0"))
                state[f"{block_attr}.shortcut.1.running_var"]  = torch.tensor(_get_weights(sc_bn_name, "moving_variance:0"))

                print(f"  ✓ {block_attr} converted")

            # ── Classifier head ───────────────────────────────────────────────
            state["classifier.weight"] = torch.tensor(
                _get_weights("dense", "kernel:0").T  # Keras: (in, out) → PyTorch: (out, in)
            )
            state["classifier.bias"] = torch.tensor(_get_weights("dense", "bias:0"))
            print("  ✓ Classifier head converted")

        except KeyError as e:
            print(f"\nWARNING: {e}")
            print(
                "\nThe HDF5 layer names do not match the expected mapping.\n"
                "This can happen with different releases of face_classification.\n"
                "Saving a randomly-initialised model instead.\n"
                "The pipeline will still run but emotion scores will be random.\n"
                "To fix: re-train with 'python scripts/train_fer.py' (coming soon)."
            )

    model.load_state_dict(state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "labels": [
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    ]}, output_path)
    print(f"\n✅ Weights saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


def inspect_hdf5(hdf5_path: Path) -> None:
    """Print the HDF5 file structure (useful for debugging layer name mapping)."""
    import h5py

    def _print_keys(grp, indent=0):
        for key in grp.keys():
            item = grp[key]
            prefix = "  " * indent
            if hasattr(item, "shape"):
                print(f"{prefix}{key}: {item.shape}")
            else:
                print(f"{prefix}[{key}]")
                _print_keys(item, indent + 1)

    with h5py.File(hdf5_path, "r") as f:
        print("HDF5 structure:")
        _print_keys(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert Mini-Xception FER2013 weights to PyTorch."
    )
    parser.add_argument(
        "--output", "-o",
        default=str(OUTPUT_PATH),
        help=f"Output .pt file path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print the HDF5 file structure and exit (useful for debugging).",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading if HDF5 already exists locally.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download HDF5
    if HDF5_CACHE.exists() and args.skip_download:
        print(f"Using cached HDF5: {HDF5_CACHE}")
    else:
        download_hdf5(HDF5_URL, HDF5_CACHE)

    # Debug mode: inspect and exit
    if args.debug:
        inspect_hdf5(HDF5_CACHE)
        return

    # Step 2: Convert to PyTorch
    convert_hdf5_to_pytorch(HDF5_CACHE, output_path)


if __name__ == "__main__":
    main()
