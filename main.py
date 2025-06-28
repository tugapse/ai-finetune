# Import the LLMTrainer class from its separate file (llm_trainer_class.py in practice)
from llm_trainer_class import LLMTrainer

# Import the datasets from their separate file (llm_datasets.py in practice)
from llm_datasets import training_dataset, test_dataset
import json


if __name__ == "__main__":
    # Instantiate the trainer
    # Choose a model you know your system can handle (Gemma 2B is safest for 8GB VRAM)
    # Be aware: 'training_dataset' in llm_datasets.py is currently a dummy for demonstration.
    # For real fine-tuning, you'd need a much larger, diverse dataset here.
    trainer_instance = LLMTrainer(
        model_name="google/gemma-2b", # Or "mistralai/Mistral-7B-v0.1" if feeling adventurous with 8GB VRAM
        train_dataset=training_dataset,
        output_dir="./fine_tuned_llm_adapters",
        num_train_epochs=1, # Keep small for quick test; increase for better results
        logging_steps=100,
    )

    # Run the fine-tuning
    trainer_instance.train()

    # Save the fine-tuned adapters
    trainer_instance.save_model()

    print("\n--- Test Dataset ---")
    # You can now use the `test_dataset` to evaluate your fine-tuned model manually
    # or with separate evaluation scripts/frameworks.
    for i, entry in enumerate(test_dataset):
        print(f"\nTest Entry {i+1}:")
        for turn in entry["turns"]:
            print(f"{turn['role'].capitalize()}: {turn['content']}")
            if turn["role"] == "assistant" and "tool_call" in turn and turn["tool_call"] is not None:
                print(f"  (Expected Tool Call: {json.dumps(turn['tool_call'])})")
                print(f"  (Expected Tool Output: {turn.get('tool_output', 'N/A')})")

    print("\nTraining and dataset preparation complete. You can now use the saved adapters for inference.")
