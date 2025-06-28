import torch
import json
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Import FastLanguageModel from Unsloth for optimized QLoRA
# Ensure Unsloth is installed: pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" (adjust cu121 for your CUDA version)
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Unsloth not found. Please install it for optimal performance on low VRAM.")
    print('pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"')
    # Fallback to standard Hugging Face components if Unsloth is not available
    # Note: This might lead to Out-Of-Memory (OOM) errors on 8GB VRAM for 7B models.
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Added BitsAndBytesConfig here
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

        self._prepare_model_and_tokenizer()
        self._format_dataset()

    def _prepare_model_and_tokenizer(self):
        """
        Loads the base model and tokenizer, applying QLoRA and Unsloth optimizations.
        """
        print(f"Loading model: {self.model_name} with QLoRA...")

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
            print("Unsloth and QLoRA setup complete.")
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
            self.tokenizer.pad_token = self.tokenizer.eos_token # Important for Llama/Gemma
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
            print("Standard Hugging Face QLoRA setup complete.")


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
        print("Formatting training dataset...")
        hf_dataset = Dataset.from_list(self.raw_train_dataset)
        self.formatted_train_dataset = hf_dataset.map(
            self._format_conversation,
            remove_columns=hf_dataset.column_names, # Remove original columns to keep only 'text'
            num_proc=4, # Use multiple processes for faster mapping if your CPU has cores
        )
        print("Dataset formatting complete.")

    def train(self):
        """
        Starts the fine-tuning process.
        """
        print("Starting LLM fine-tuning...")

        # Default training arguments for low VRAM
        # These are crucial and should be carefully tuned based on your specific GPU
        # If you experience OOM, increase gradient_accumulation_steps or decrease max_seq_length.
        default_training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3, # Start with a small number of epochs (e.g., 3-5)
            per_device_train_batch_size=1, # Smallest possible batch size
            gradient_accumulation_steps=4, # Simulate a larger batch size (1 * 4 = effective batch size of 4)
            warmup_steps=10, # Number of warmup steps for learning rate scheduler
            learning_rate=2e-4, # Common learning rate for QLoRA
            fp16=not torch.cuda.is_bf16_supported(), # Use FP16 if bfloat16 is not supported
            bf16=torch.cuda.is_bf16_supported(), # Use bfloat16 if GPU supports it (Nvidia RTX 30 series+)
            logging_steps=1, # Log progress frequently
            optim="adamw_8bit", # Use 8-bit AdamW for memory efficiency
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            # Uncomment and adjust if you want to save checkpoints
            # save_steps=500,
            # save_total_limit=3,
            push_to_hub=False, # Set to True to push model to Hugging Face Hub (requires login)
        )

        # Override defaults with any user-provided arguments
        final_training_args = TrainingArguments(
            **default_training_args.to_dict(),
            **self.training_kwargs
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.formatted_train_dataset,
            args=final_training_args,
            packing=False, # Set to True if your dataset has many short examples for efficiency
            max_seq_length=2048, # Must match model loading max_seq_length
        )

        trainer.train()
        print("Fine-tuning complete!")

    def save_model(self, save_path: str = None):
        """
        Saves the fine-tuned LoRA adapters.

        Args:
            save_path (str, optional): The path to save the adapters. Defaults to self.output_dir.
        """
        final_save_path = save_path if save_path else self.output_dir
        print(f"Saving fine-tuned adapters to {final_save_path}...")
        self.model.save_pretrained_merged(final_save_path, self.tokenizer, save_method="merged_4bit")
        print("Model adapters saved successfully.")
