# LLM Fine-tuning Environment Setup

## 1. Install System Dependencies (if not already present)
Ensure you have essential development tools and CUDA drivers installed. For Arch Linux, this typically involves:
```bash
sudo pacman -S --needed base-devel cuda opencl-headers opencl-nvidia # For NVIDIA GPUs
# Or for AMD GPUs (check specific drivers for your card)
# sudo pacman -S --needed base-devel rocm-hip-sdk rocm-opencl-runtime
```
Make sure your NVIDIA drivers are up to date and compatible with the CUDA toolkit version you plan to use.

## 2. Set Up a Python Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to avoid conflicts with your system's Python packages.
```bash
sudo pacman -S python-pip python-venv # Install pip and venv if not already installed
python3 -m venv llm_finetune_env
source llm_finetune_env/bin/activate
```
You will see `(llm_finetune_env)` prepended to your terminal prompt, indicating that the virtual environment is active.

## 3. Install PyTorch with CUDA Support
This is the most crucial step and depends on your CUDA toolkit version.
Go to the official PyTorch website: https://pytorch.org/get-started/locally/
Select your operating system: Linux  
Package Manager: Pip  
CUDA Version: Choose the CUDA version that matches your installed NVIDIA CUDA toolkit. For example, if you have CUDA 12.1, select CUDA 12.1.  
You will get a command like this (example for CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Execute this specific command in your active virtual environment.

## 4. Install Other Core Dependencies from requirements.txt
Make sure you have saved the provided requirements.txt file in your project directory.
```bash
pip install -r requirements.txt
```
This command will install transformers, peft, bitsandbytes, trl, accelerate, and datasets.

## 5. Install Unsloth (Crucial for Low VRAM Optimization)
Unsloth needs a specific installation command that links it to your CUDA version. It's provided in your requirements.txt as commented-out lines.
Uncomment the line that matches your CUDA version in requirements.txt or run it separately:
For CUDA 12.1:
```bash
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```
For CUDA 11.8:
```bash
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```
(If your CUDA version is different, refer to Unsloth's official documentation for the correct installation string.)

## 6. Verify Installation
You can quickly check if PyTorch is recognizing your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```
This should output `True`, the number of detected GPUs, and the name of your first GPU.

## 7. Run Your Fine-tuning Script
Once all dependencies are installed, you can run your `run_training_script.py`:
```bash
python run_training_script.py
```

## Important Notes for Arch Linux
- **CUDA Toolkit**: Ensure your installed CUDA toolkit version matches the PyTorch and Unsloth versions you're installing. Mismatches are a common source of errors.
- **NVIDIA Drivers**: Keep your NVIDIA drivers updated.
- **Virtual Environments**: Always activate your virtual environment (`source llm_finetune_env/bin/activate`) before installing packages or running your script.
- **Disk Space**: LLMs and their training data can consume significant disk space. Ensure you have ample free space on an SSD.
- **VRAM Monitoring**: Use `nvidia-smi` in a separate terminal during training to monitor your GPU's VRAM usage. This will help you adjust hyperparameters if you encounter out-of-memory errors.
