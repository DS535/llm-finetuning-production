"""
Automated setup script for Google Colab environment.

Run this at the beginning of each Colab session to:
- Mount Google Drive
- Clone GitHub repository
- Install dependencies
- Configure authentication
"""

import os
import subprocess
import sys


def mount_google_drive():
    """Mount Google Drive."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
        return True
    except ImportError:
        print("✗ Not running in Colab environment")
        return False
    except Exception as e:
        print(f"✗ Failed to mount Google Drive: {e}")
        return False


def check_gpu():
    """Check if GPU is available and print info."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\n=== GPU Information ===")
        print(result.stdout)

        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ Total GPU Memory: {total_mem:.2f} GB")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False


def install_dependencies(requirements_path: str = "requirements.txt"):
    """Install dependencies from requirements.txt."""
    if not os.path.exists(requirements_path):
        print(f"✗ Requirements file not found: {requirements_path}")
        return False

    try:
        print("\n=== Installing Dependencies ===")
        print("This may take 5-10 minutes...")

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
            check=True
        )

        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def clone_github_repo(repo_url: str, target_dir: str = "llm-finetuning-production"):
    """Clone GitHub repository."""
    if os.path.exists(target_dir):
        print(f"✓ Repository already exists at: {target_dir}")
        return True

    try:
        print(f"\n=== Cloning Repository ===")
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
        print(f"✓ Repository cloned to: {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return False


def setup_wandb(api_key: str = None):
    """Setup Weights & Biases for experiment tracking."""
    try:
        import wandb

        if api_key:
            wandb.login(key=api_key)
            print("✓ W&B configured with API key")
        else:
            print("! W&B API key not provided. Set with: wandb.login(key='YOUR_KEY')")

        return True
    except ImportError:
        print("✗ wandb not installed")
        return False
    except Exception as e:
        print(f"✗ W&B setup failed: {e}")
        return False


def setup_huggingface(token: str = None):
    """Setup HuggingFace authentication."""
    try:
        from huggingface_hub import login

        if token:
            login(token=token)
            print("✓ HuggingFace configured with token")
        else:
            print("! HF token not provided. Set with: huggingface_hub.login(token='YOUR_TOKEN')")

        return True
    except ImportError:
        print("✗ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"✗ HuggingFace setup failed: {e}")
        return False


def full_setup(
    repo_url: str = None,
    wandb_key: str = None,
    hf_token: str = None
):
    """
    Run complete Colab setup.

    Args:
        repo_url: GitHub repository URL
        wandb_key: Weights & Biases API key
        hf_token: HuggingFace access token
    """
    print("=" * 60)
    print("LLM Finetuning Production - Colab Setup")
    print("=" * 60)

    # Step 1: Mount Drive
    mount_google_drive()

    # Step 2: Check GPU
    check_gpu()

    # Step 3: Clone repo (if URL provided)
    if repo_url:
        clone_github_repo(repo_url)
        os.chdir("llm-finetuning-production")

    # Step 4: Install dependencies
    install_dependencies()

    # Step 5: Setup W&B
    if wandb_key:
        setup_wandb(wandb_key)

    # Step 6: Setup HuggingFace
    if hf_token:
        setup_huggingface(hf_token)

    print("\n" + "=" * 60)
    print("✓ Setup complete! Ready to start training.")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage in Colab:
    # from setup_colab import full_setup
    # full_setup(
    #     repo_url="https://github.com/YOUR_USERNAME/llm-finetuning-production.git",
    #     wandb_key="YOUR_WANDB_KEY",
    #     hf_token="YOUR_HF_TOKEN"
    # )
    full_setup()
