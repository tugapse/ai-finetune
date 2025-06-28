

# Local LLM Fine-tuning for Tool Use

This repository provides a robust framework for fine-tuning Large Language Models (LLMs) to effectively utilize external tools (function calling) directly on consumer-grade GPUs, even with limited VRAM (e.g., 8GB). It leverages advanced techniques like QLoRA and Unsloth to optimize for memory and speed, making powerful AI customization accessible locally.✨

## Features

- **QLoRA-Powered Fine-tuning**: Efficiently fine-tune large models by only training a small number of parameters, drastically reducing VRAM requirements.
- **Unsloth Integration**: Utilizes Unsloth for highly optimized and faster LoRA/QLoRA training on consumer NVIDIA GPUs.
- **Conversational Data Format**: Supports training data structured as multi-turn conversations, crucial for realistic tool-use scenarios and conversational AI.
- **Modular Design**: Separate Python classes for the trainer and datasets, promoting clean code and reusability.
- **Customizable Tool Schema**: Easily define and integrate your own tool definitions for specialized applications.
- **Optimized for Low VRAM**: Includes configurations for gradient accumulation and gradient checkpointing to manage memory effectively.

## 🚀 Getting Started

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (e.g., RTX 3050, RTX 2060 Super, some GTX 1080/1080 Ti). More VRAM is always better, but 8GB is the target with careful optimization.
- **System RAM**: At least 16GB, preferably 32GB+ for smoother data handling.
- **Storage**: Ample SSD space (100GB+ recommended) for models, datasets, and checkpoints.

### Installation

Detailed step-by-step installation instructions for Arch Linux, including virtual environment setup and specific commands for PyTorch and Unsloth (compatible with CUDA 12.x), can be found in our dedicated installation guide.

**Please refer to the [Installation Instructions for LLM Fine-tuning on Arch Linux](INSTALLATION_GUIDE.md) document for complete setup details.**

### Summary of Key Installation Steps

- **Cleanup**: Clear previous virtual environments and pip cache.
- **System Dependencies**: Install base-devel, cuda, opencl-headers, opencl-nvidia (for NVIDIA).
- **Python Virtual Environment**: Create and activate a python3.11 virtual environment.
- **PyTorch (CUDA 12.1)**: Install the latest stable PyTorch for CUDA 12.1 from the official PyTorch website.
- **Unsloth (CUDA 12.1)**: Install Unsloth, specifying [cu121] for CUDA 12.1 compatibility.
- **Other Dependencies**: Install transformers, peft, bitsandbytes, trl, accelerate, datasets from requirements.txt.
- **Verify**: Confirm all installations and CUDA detection are correct.

## Project Structure

```
├── llm_trainer_class.py      # Defines the LLMTrainer class
├── llm_datasets.py           # Contains training and test datasets
├── run_training_script.py    # Main script to run the fine-tuning process
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

## 🏃‍♀️ Usage

1. **Prepare Your Data**  
   Ensure your training data is in the specified conversational JSON format (see "Dataset Format" below). You can define your training_dataset in llm_datasets.py or load it from an external JSON file.

2. **Configure and Run the Trainer**  
   Edit `run_training_script.py` to select your desired base model (e.g., "google/gemma-2b") and adjust training arguments as needed.

   ```python
   # run_training_script.py (excerpt)
   from llm_trainer_class import LLMTrainer
   from llm_datasets import training_dataset, test_dataset

   if __name__ == "__main__":
       trainer_instance = LLMTrainer(
           model_name="google/gemma-2b", # Recommended for 8GB VRAM
           train_dataset=training_dataset,
           output_dir="./fine_tuned_llm_adapters",
           num_train_epochs=3, # Example: Adjust as needed
           logging_steps=100,
           # ... other training arguments
       )
       trainer_instance.train()
       trainer_instance.save_model()
   ```

   Then, run the script from your terminal:
   ```bash
   python run_training_script.py
   ```

3. **Evaluate (Post-Training)**  
   After fine-tuning, your model's adapters will be saved to the specified `output_dir`. You can then load these adapters with the base model for inference and evaluate its performance on your `test_dataset`.

## 📚 Dataset Format

The training and test datasets should follow a conversational JSON structure, demonstrating both standard responses and tool-calling behavior:

```json
[
  {
    "conversation_id": "unique_id_1",
    "turns": [
      {"role": "user", "content": "User's query here."},
      {
        "role": "assistant",
        "tool_call": {
          "tool_name": "example_tool",
          "parameters": {"param1": "value1", "param2": "value2"}
        },
        "tool_output": "Output from the example_tool.",
        "content": "Assistant's natural language response incorporating tool output."
      }
    ]
  },
  {
    "conversation_id": "unique_id_2",
    "turns": [
      {"role": "user", "content": "Another user query."},
      {"role": "assistant", "content": "A direct assistant response without a tool call."}
    ]
  }
]
```

The `tool_call` and `tool_output` fields are crucial for training the LLM to understand when to invoke a tool and how to integrate its results.

## ⚠️ Troubleshooting

### CUDA Out Of Memory Errors

- **Increase gradient_accumulation_steps**: In `run_training_script.py`, set `gradient_accumulation_steps` higher (e.g., 8, 16, or 32). This simulates a larger batch size over multiple steps, reducing peak VRAM usage.
- **Decrease max_seq_length**: Reduce the `max_seq_length` in both `FastLanguageModel.from_pretrained` and `SFTTrainer` arguments. This limits the context window but saves significant VRAM.
- **Ensure use_gradient_checkpointing=True**: This is enabled by default in the `LLMTrainer` class but verify it's active.
- **Close other GPU applications**: Browsers, games, or other AI tools can consume VRAM.

### Dependency Conflicts (ResolutionImpossible)

This is often due to Python/CUDA/PyTorch/Unsloth version mismatches. Re-run all steps from the "0. Crucial Pre-Installation Cleanup" section of the installation guide. Ensure you precisely match PyTorch installation to your system's CUDA version and then install Unsloth for the corresponding CUDA version.

### Slow Training

- **Ensure Unsloth is correctly installed and active**: `_UNSLOTH_AVAILABLE` should be `True` in `llm_trainer_class.py` during runtime.
- **Your GPU might simply be at its limits**: Consider using a smaller model or reducing dataset size for faster iteration.

## 🤝 Contributing

Contributions are welcome! If you find bugs, have suggestions, or want to add new features (e.g., support for other PEFT methods, more robust evaluation scripts), please open an issue or submit a pull request.