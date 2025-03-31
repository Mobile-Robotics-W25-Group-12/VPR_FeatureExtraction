# === Environment Sanity Check Script ===
# Purpose: Test basic import and GPU functionality for packages
#          in the vprtutorial2 environment (TensorFlow removed).
# Usage: conda activate vprtutorial2 && python your_script_name.py

import sys
import importlib
import logging
import time
import os
import traceback  # Moved import here as it's only used in run_test's except block

# --- Global Variables ---
results = {"success": [], "failure": [], "warning": []}


# --- Helper Function ---
def run_test(name, test_func):
    """Runs a test function, captures results."""
    print(f"--- Testing: {name} ---")
    try:
        test_func()
        print(f"✅ Success: {name}\n")
        results["success"].append(name)
    except ImportError as e:
        print(f"❌ ImportError: {name} - Package not found or import failed. {e}\n")
        results["failure"].append(f"{name} (ImportError)")
    except Exception as e:
        print(f"❌ Failure: {name} - Test execution failed:")
        print(traceback.format_exc())  # Print full traceback for runtime errors
        print("")
        results["failure"].append(f"{name} (RuntimeError)")  # Simplified summary


# --- Test Functions ---


def test_python_version():
    """Checks Python Version"""
    print(f"Python version: {sys.version}")
    # Basic check, always succeeds if script runs
    if sys.version_info.major < 3:
        raise RuntimeError("Python 3 is required.")


def test_numpy():
    """Tests NumPy basic array operations."""
    import numpy as np

    print(f"NumPy version: {np.__version__}")
    a = np.array([1, 2, 3])
    b = np.ones((2, 2))
    c = np.sum(b)
    assert c == 4.0
    print(f"NumPy check: array shape {a.shape}, op result {c}")


def test_pytorch():
    """Tests PyTorch import, CUDA availability, and basic GPU op."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"PyTorch CUDA device count: {device_count}")
        if device_count > 0:
            try:
                device_name = torch.cuda.get_device_name(0)
                print(f"Using device 0: {device_name}")
                device = torch.device("cuda:0")
                # Perform a simple operation on GPU
                a = torch.randn(3, 3, device=device)
                b = torch.randn(3, 3, device=device)
                c = a @ b  # Matrix multiplication
                _ = c.cpu()  # Sync and check if op ran without error
                print(f"PyTorch GPU tensor op completed successfully.")
            except Exception as e:
                raise RuntimeError(f"PyTorch GPU operation failed: {e}") from e
        else:
            results["warning"].append("PyTorch (CUDA available but device_count=0)")
            print("⚠️ Warning: PyTorch reports CUDA available but finds 0 devices.")
    else:
        results["warning"].append("PyTorch (CUDA not available)")
        print(
            "⚠️ Warning: PyTorch cannot find/use CUDA. CPU execution should still work."
        )


def test_torchvision():
    """Tests Torchvision import."""
    import torchvision

    print(f"Torchvision version: {torchvision.__version__}")
    # Basic import is often sufficient
    # from torchvision.models import resnet18 # Could try loading a model definition
    # model = resnet18()
    print("Torchvision imported successfully.")


# === test_tensorflow function removed ===


def test_faiss():
    """Tests FAISS CPU index and GPU index creation/ops."""
    # Needs numpy
    import numpy as np
    import faiss

    print(f"FAISS version: {faiss.__version__}")

    # Test CPU basic functionality first
    try:
        d_cpu = 32
        index_cpu = faiss.IndexFlatL2(d_cpu)
        print(
            f"FAISS CPU index created (is_trained={index_cpu.is_trained}, ntotal={index_cpu.ntotal})"
        )
    except Exception as e:
        # If even CPU fails, make it a hard failure for faiss
        raise RuntimeError(f"FAISS CPU basic test failed: {e}") from e

    # Test GPU functionality
    num_gpus = faiss.get_num_gpus()
    print(f"FAISS detected {num_gpus} GPU(s).")
    if num_gpus > 0:
        try:
            res = faiss.StandardGpuResources()  # Attempt to initialize GPU resources
            print(
                f"FAISS StandardGpuResources initialized successfully on default device."
            )
            # Simple index test
            d = 64  # dimension
            nb = 100  # database size (keep small for test)
            xb = np.random.rand(nb, d).astype("float32")
            index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  # move to GPU 0
            gpu_index_flat.add(xb)  # add vectors to the index
            assert (
                gpu_index_flat.ntotal == nb
            ), f"Expected {nb} vectors, found {gpu_index_flat.ntotal}"
            print(f"FAISS GPU index created and added {gpu_index_flat.ntotal} vectors.")
            # Test search (optional but good)
            xq = np.random.rand(5, d).astype("float32")
            D, I = gpu_index_flat.search(xq, 4)  # search 4 nearest neighbors
            assert I.shape == (
                5,
                4,
            ), f"Expected search result shape (5, 4), got {I.shape}"
            print(f"FAISS GPU index search successful.")

        except Exception as e:
            raise RuntimeError(f"FAISS GPU resource/index test failed: {e}") from e
    else:
        results["warning"].append("FAISS (No GPU detected)")
        print(
            "⚠️ Warning: FAISS did not detect any GPUs. CPU execution should still work."
        )


def test_opencv():
    """Tests OpenCV import and basic image operation."""
    # Needs numpy
    import numpy as np

    # OpenCV is imported as cv2
    import cv2

    print(f"OpenCV version: {cv2.__version__}")
    img = np.zeros((50, 50, 3), dtype=np.uint8)  # Create dummy image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (50, 50)
    print(f"OpenCV basic image operation successful.")
    # Optional: Check CUDA build info (doesn't guarantee runtime)
    try:
        build_info = cv2.getBuildInformation()
        cuda_section = [s for s in build_info.split("\n\n") if "CUDA" in s]
        if cuda_section and "YES" in cuda_section[0]:
            print(
                "   Build Information indicates OpenCV was compiled with CUDA support."
            )
        else:
            print(
                "   Build Information indicates OpenCV was NOT compiled with CUDA support."
            )
    except Exception:
        print("   Could not retrieve OpenCV build information.")


def test_pillow():
    """Tests Pillow import and basic image creation."""
    # Pillow is imported from PIL
    from PIL import Image

    print(f"Pillow version: {Image.__version__}")
    img = Image.new("RGB", (60, 30), color="red")
    assert img.size == (60, 30)
    print(f"Pillow basic image creation successful.")


def test_sklearn():
    """Tests Scikit-learn import and model instantiation."""
    # Scikit-learn is imported as sklearn
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print(f"Scikit-learn version: {sklearn.__version__}")
    model = LogisticRegression()
    scaler = StandardScaler()
    print("Scikit-learn basic model/transformer instantiation successful.")


def test_skimage():
    """Tests Scikit-image import and basic filter operation."""
    # Needs numpy
    import numpy as np

    # Scikit-image is imported as skimage
    import skimage
    from skimage import filters
    from skimage.data import camera  # Use a built-in image

    print(f"Scikit-image version: {skimage.__version__}")
    image = camera()
    edges = filters.sobel(image)
    assert edges.shape == image.shape
    print(f"Scikit-image filter applied successfully.")


def test_matplotlib():
    """Tests Matplotlib import."""
    # Matplotlib-base means we mainly care about the core and pyplot
    import matplotlib

    print(f"Matplotlib version: {matplotlib.__version__}")
    # Importing pyplot is a decent check for base install issues
    import matplotlib.pyplot as plt

    print("Matplotlib.pyplot imported successfully.")


# === test_tfhub function removed ===


def test_patchnetvlad():
    """Tests PatchNetVLAD import (basic)."""
    # Assuming the package name to import is 'patchnetvlad'
    # Replace with actual import name if different
    try:
        module = importlib.import_module("patchnetvlad")
        # Check for version if available
        version = getattr(module, "__version__", "N/A")
        print(f"PatchNetVLAD version: {version}")
        print("PatchNetVLAD imported successfully (basic check).")
        # TODO: Add a more specific call if a simple function/class is known
        # e.g., model = module.SomeModelClass()
    except ModuleNotFoundError:
        results["warning"].append("PatchNetVLAD (ModuleNotFoundError)")
        print(
            "⚠️ Warning: Could not find module named 'patchnetvlad'. Is it installed and named correctly?"
        )
    except ImportError as e:
        raise RuntimeError(f"PatchNetVLAD import failed: {e}") from e


def test_natsort():
    """Tests natsort basic sorting."""
    import natsort

    print(f"natsort version: {natsort.__version__}")
    data = ["version10", "version2", "version1"]
    expected = ["version1", "version2", "version10"]
    sorted_data = natsort.natsorted(data)
    assert sorted_data == expected, f"Expected {expected}, got {sorted_data}"
    print(f"natsort sorting successful: {data} -> {sorted_data}")


def test_tqdm():
    """Tests tqdm progress bar basic function."""
    import tqdm
    from time import sleep

    print(f"tqdm version: {tqdm.__version__}")
    total = 0
    # Use file=sys.stdout for better compatibility across terminals
    for i in tqdm.tqdm(
        range(3),
        desc="tqdm test",
        file=sys.stdout,
        ncols=80,
        bar_format="{l_bar}{bar}|",
    ):
        sleep(0.05)
        total += i
    print(f"\ntqdm loop finished (total={total}).")  # Add newline after progress bar


def test_ipywidgets():
    """Tests ipywidgets import."""
    import ipywidgets

    print(f"ipywidgets version: {ipywidgets.__version__}")
    # Just check import, functionality requires Jupyter context
    # w = ipywidgets.IntSlider() # Instantiation should work
    # print("ipywidgets basic instantiation successful.")
    print("ipywidgets imported successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Environment Sanity Check (No TensorFlow) ---")  # Updated title slightly
    start_time = time.time()

    # Map YAML name / common name to test function
    # Order roughly based on YAML, grouping related items
    test_suite = {
        "Python Env": test_python_version,
        "NumPy": test_numpy,
        "PyTorch": test_pytorch,
        "Torchvision": test_torchvision,
        # "TensorFlow": test_tensorflow, # Removed
        # "TensorFlow Hub": test_tfhub, # Removed
        "FAISS": test_faiss,  # Covers faiss-gpu
        "OpenCV": test_opencv,  # Covers opencv
        "Pillow": test_pillow,
        "Scikit-learn": test_sklearn,  # Covers scikit-learn
        "Scikit-image": test_skimage,  # Covers scikit-image
        "Matplotlib": test_matplotlib,  # Covers matplotlib-base
        "PatchNetVLAD": test_patchnetvlad,
        "natsort": test_natsort,
        "tqdm": test_tqdm,
        "ipywidgets": test_ipywidgets,
    }

    # Run all tests
    for name, func in test_suite.items():
        run_test(name, func)

    # --- Summary ---
    print("\n--- Test Summary ---")
    duration = time.time() - start_time
    print(f"Total time: {duration:.2f} seconds")
    print(f"✅ Success: {len(results['success'])}")
    if results["success"]:
        print(f"   Packages/Checks: {', '.join(results['success'])}")

    print(f"⚠️ Warnings: {len(results['warning'])}")
    if results["warning"]:
        print(f"   Details: {', '.join(results['warning'])}")

    print(f"❌ Failures: {len(results['failure'])}")
    if results["failure"]:
        print(f"   Details: {', '.join(results['failure'])}")

    print("--------------------")

    if results["failure"]:
        print("\n🔴 Some essential tests failed. Please review the errors above.")
        sys.exit(1)  # Exit with error code 1 if failures occurred
    elif results["warning"]:
        print(
            "\n🟡 Some tests passed with warnings (e.g., GPU not found/used). Review warnings if GPU usage is critical."
        )
        sys.exit(0)  # Exit with code 0, but indicate warnings occurred
    else:
        print("\n🟢 All essential tests passed successfully!")
        sys.exit(0)  # Exit with code 0 for success
