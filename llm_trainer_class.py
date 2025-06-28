import torch
import json
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import os
from colors_util import Color, pformat_text # Import Color and pformat_text

# Import FastLanguageModel from Unsloth for optimized QLoRA
# Ensure Unsloth is installed: pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" (adjust cu121 for your CUDA version)
try:
    from unsloth import FastLanguageModel
except ImportError:
    pformat_text("Unsloth not found. Please install it for optimal performance on low VRAM.", Color.YELLOW, Color.BOLD)
    pformat_text('To install: pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" (adjust for your CUDA)', Color.NORMAL_YELLOW)
    # Fallback to standard Hugging Face components if Unsloth is not available
    # Note: This might lead to Out-Of-Memory (OOM) errors on 8GB VRAM for 7B models.
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _UNSLOTH_AVAILABLE = False
else:
    _UNSLOTH_AVAILABLE = True


class LLMTrainer:
    """
    A trainer class for fine-tuning Large Language Models using QLoRA,
    optimized for low VRAM environments like laptops (e.g., 8GB VRAM).

    It utilizes Unsloth for speed and memory efficiency if available,
    and incorporates QLoRA, gradient accumulation, and gradient checkpointing.
    """
    def __init__(self, model_name: str, train_dataset: list, output_dir: str, **kwargs):
        """
        Initializes the LLMTrainer.

        Args:
            model_name (str): The name of the pre-trained model from Hugging Face Hub (e.g., "google/gemma-2b").
            train_dataset (list): The training data in conversational format (list of dicts).
            output_dir (str): Directory to save fine-tuned model adapters.
            **kwargs: Additional arguments for TrainingArguments (e.g., num_train_epochs, learning_rate).
        """
        self.model_name = model_name
        self.raw_train_dataset = train_dataset
        self.output_dir = output_dir
        self.training_kwargs = kwargs

        self.model = None
        self.tokenizer = None
        self.formatted_train_dataset = None

        # Define tool schema (this would typically be part of your system prompt)
        self.tool_schema = """
        Available tools:
        - get_weather(location: str, unit: str = "celsius"): Retrieves current weather conditions.
        - search_web(query: str): Performs a web search.
        - perform_calculation(expression: str): Evaluates a mathematical expression.
        - schedule_event(title: str, date: str, time: str, attendees: list = []): Schedules a calendar event.
        """

        # For multiprocessing safety with CUDA
        # This is a common pattern for libraries like datasets and torch.
        # It's recommended to set this at the very beginning of your main script.
        # However, for simplicity here, we'll try to enforce it before dataset mapping.
        if "OMP_NUM_THREADS" not in os.environ:
             # Reduce number of OpenMP threads to avoid too many CPU threads on smaller systems.
            os.environ["OMP_NUM_THREADS"] = "1"
        if torch.cuda.is_available() and os.name == 'posix': # Check if CUDA is available and OS is Linux/Unix
            # Set multiprocessing start method for CUDA compatibility
            # This must be done ONCE at the very beginning of the script execution (ideally in main.py)
            # If done here, it might still cause issues if other libraries initialize CUDA before.
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
                pformat_text("Multiprocessing start method set to 'spawn' for CUDA compatibility.", Color.NORMAL_CYAN)
            except RuntimeError:
                pformat_text("Multiprocessing start method already set. Skipping.", Color.NORMAL_YELLOW)


        self._prepare_model_and_tokenizer()
        self._format_dataset()

    def _prepare_model_and_tokenizer(self):
        """
        Loads the base model and tokenizer, applying QLoRA and Unsloth optimizations.
        """
        pformat_text(f"Loading model: {self.model_name} with QLoRA...", Color.BLUE, Color.BOLD)

        if _UNSLOTH_AVAILABLE:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=2048, # Adjust based on average conversation length
                dtype=None,  # Auto-detects bfloat16 or float16
                load_in_4bit=True,  # Enables QLoRA
            )

            # Prepare model for LoRA training
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank: common values are 8, 16, 32. Lower saves VRAM, higher captures more detail.
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16, # LoRA alpha: typically same as r or 2*r
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=True,  # CRITICAL for 8GB VRAM
                random_state=42,
                max_seq_length=2048, # Must match above
            )
            pformat_text("Unsloth and QLoRA setup complete.", Color.GREEN, Color.BOLD)
        else:
            # Fallback for standard Hugging Face (might be less VRAM efficient)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure tokenizer has a pad_token if not present, crucial for batching
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token # Common for CausalLMs
            self.tokenizer.padding_side = "right" # Or "left", depending on model/task

            self.model = prepare_model_for_kbit_training(self.model)
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=16,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.gradient_checkpointing_enable()
            pformat_text("Standard Hugging Face QLoRA setup complete.", Color.GREEN, Color.BOLD)


    def _format_conversation(self, example: dict) -> dict:
        """
        Formats a single conversation entry into the prompt-response text
        expected by the LLM for fine-tuning. This includes the tool schema
        and structured tool calls/outputs.

        Args:
            example (dict): A single conversation entry from the raw dataset.

        Returns:
            dict: A dictionary with a "text" key containing the formatted string.
        """
        formatted_text = f"### System:\n{self.tool_schema.strip()}\n\n"
        for turn in example["turns"]:
            if turn["role"] == "user":
                formatted_text += f"### User:\n{turn['content']}\n\n"
            elif turn["role"] == "assistant":
                if "tool_call" in turn and turn["tool_call"] is not None:
                    # Convert tool_call dict to JSON string for the LLM to learn to generate
                    tool_call_json = json.dumps(turn["tool_call"], indent=2)
                    tool_output_str = f"Tool output: {turn['tool_output']}\n" if "tool_output" in turn else ""
                    formatted_text += f"### Assistant:\n{tool_call_json}\n{tool_output_str}{turn['content']}\n\n"
                else:
                    formatted_text += f"### Assistant:\n{turn['content']}\n\n"
        return {"text": formatted_text.strip()}

    def _format_dataset(self):
        """
        Applies the formatting function to the entire raw training dataset.
        """
        pformat_text("Formatting training dataset...", Color.BLUE)
        hf_dataset = Dataset.from_list(self.raw_train_dataset)
        self.formatted_train_dataset = hf_dataset.map(
            self._format_conversation,
            remove_columns=hf_dataset.column_names, # Remove original columns to keep only 'text'
            num_proc=1, # Changed from 4 to 1 to disable multiprocessing and avoid CUDA re-init error
        )
        pformat_text("Dataset formatting complete.", Color.GREEN)

    def train(self):
        """
        Starts the fine-tuning process.
        """
        pformat_text("Starting LLM fine-tuning...", Color.BLUE, Color.BOLD)

        # Define default training arguments as a dictionary
        default_args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": 3, # Default epoch count
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 10,
            "learning_rate": 2e-4,
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 42,
            # Uncomment and adjust if you want to save checkpoints
            # "save_steps": 500,
            # "save_total_limit": 3,
            "push_to_hub": False,
        }

        # Merge user-provided kwargs, allowing them to override defaults
        final_args = {**default_args_dict, **self.training_kwargs}

        final_training_args = TrainingArguments(**final_args) # Unpack the merged dictionary


        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.formatted_train_dataset,
            args=final_training_args,
            packing=False,
            dataset_text_field="text", # Specify which column holds the text for training
            max_seq_length=2048, # Must match model loading max_seq_length
        )

        trainer.train()
        pformat_text("Fine-tuning complete!", Color.GREEN, Color.BOLD)

    def save_model(self, save_path: str = None):
        """
        Saves the fine-tuned LoRA adapters.

        Args:
            save_path (str, optional): The path to save the adapters. Defaults to self.output_dir.
        """
        final_save_path = save_path if save_path else self.output_dir
        pformat_text(f"Saving fine-tuned adapters to {final_save_path}...", Color.BLUE)
        if _UNSLOTH_AVAILABLE:
            # Use Unsloth's optimized save method if available
            self.model.save_pretrained_merged(final_save_path, self.tokenizer, save_method="merged_4bit")
            pformat_text("Unsloth: Model adapters saved successfully.", Color.GREEN)
        else:
            # Fallback to standard PEFT save method (saves adapters only)
            self.model.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path) # Save tokenizer with adapters
            pformat_text("Standard PEFT: Model adapters saved successfully (base model not merged).", Color.GREEN)
