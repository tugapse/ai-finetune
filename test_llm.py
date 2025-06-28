import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel # This is crucial for loading PEFT adapters
import os
from colors_util import Color, pformat_text, format_text # Import format_text as well!


# Ensure multiprocessing start method for CUDA compatibility (Crucial for GPU usage with multiprocessing)
if torch.cuda.is_available() and os.name == 'posix': # Check if CUDA is available and OS is Linux/Unix
    try:
        # This should ideally be set once at the very entry point of your application
        # If your main.py is the entry point, it's good to have it there too.
        torch.multiprocessing.set_start_method('spawn', force=True)
        pformat_text("Multiprocessing start method set to 'spawn' for CUDA compatibility.", Color.NORMAL_CYAN)
    except RuntimeError:
        pformat_text("Multiprocessing start method already set. Skipping.", Color.NORMAL_YELLOW)


# --- Configuration ---
BASE_MODEL_NAME = "google/gemma-2b" # Must match the model used for fine-tuning
ADAPTER_PATH = "./fine_tuned_llm_adapters" # Path where your LoRA adapters were saved

# Define tool schema (MUST be identical to the one used during fine-tuning)
TOOL_SCHEMA = """
Available tools:
- get_weather(location: str, unit: str = "celsius"): Retrieves current weather conditions.
- search_web(query: str): Performs a web search.
- perform_calculation(expression: str): Evaluates a mathematical method.
- schedule_event(title: str, date: str, time: str, attendees: list = []): Schedules a calendar event.
"""

# --- Load Model and Tokenizer ---
pformat_text(f"Loading base model: {BASE_MODEL_NAME}...", Color.BLUE, Color.BOLD)

# Quantization configuration for loading the base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Ensure tokenizer has a pad_token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Typically good for causal LMs

pformat_text(f"Loading LoRA adapters from: {ADAPTER_PATH}...", Color.BLUE, Color.BOLD)
# Load the PEFT adapters on top of the base model
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# You can optionally merge the adapters into the base model for potentially faster inference
# but it requires more VRAM. For 8GB, you might want to skip this or proceed cautiously.
# model = model.merge_and_unload()
# model.to("cuda") # Ensure merged model is on GPU if not already.

pformat_text("Model and adapters loaded successfully!", Color.GREEN, Color.BOLD)
pformat_text("--- Ready for Inference ---", Color.GREEN, Color.BOLD, Color.UNDERLINE)

# --- Inference Formatting Function ---
def format_inference_prompt(conversation_turns: list) -> str:
    """
    Formats the conversation turns into a prompt string for inference,
    including the tool schema.
    """
    formatted_text = f"### System:\n{TOOL_SCHEMA.strip()}\n\n"
    for turn in conversation_turns:
        if turn["role"] == "user":
            formatted_text += f"### User:\n{turn['content']}\n\n"
        elif turn["role"] == "assistant":
            # For inference, we only include the assistant's previous direct content
            # The model will generate the tool_call/tool_output parts if it needs to.
            if "tool_call" in turn and turn["tool_call"] is not None:
                tool_call_json = json.dumps(turn["tool_call"], indent=2)
                tool_output_str = f"Tool output: {turn.get('tool_output', '')}\n" # Include tool output from past turns
                formatted_text += f"### Assistant:\n{tool_call_json}\n{tool_output_str}{turn['content']}\n\n"
            else:
                formatted_text += f"### Assistant:\n{turn['content']}\n\n"
    formatted_text += "### Assistant:\n" # Prepare for the next assistant response
    return formatted_text.strip()

# --- Simulate Tool Execution (Dummy Functions) ---
def execute_tool(tool_name: str, parameters: dict) -> str:
    """
    A dummy function to simulate tool execution.
    In a real application, this would call actual APIs or functions.
    """
    pformat_text(f"[SIMULATING TOOL CALL: {tool_name} with params {parameters}]", Color.DIM)
    if tool_name == "get_weather":
        location = parameters.get("location", "unknown city")
        unit = parameters.get("unit", "celsius")
        # Dummy weather data
        if "london" in location.lower():
            return f"Weather in {location}: 18°{unit}, partly cloudy."
        elif "paris" in location.lower():
            return f"Weather in {location}: 20°{unit}, sunny."
        elif "berlin" in location.lower():
             return f"Weather in {location}: 22°{unit}, sunny."
        else:
            return f"Could not get weather for {location}."
    elif tool_name == "search_web":
        query = parameters.get("query", "empty query")
        return f"Search result for '{query}': Relevant information found online."
    elif tool_name == "perform_calculation":
        expression = parameters.get("expression", "invalid")
        try:
            result = eval(expression) # WARNING: eval() is unsafe in real apps with untrusted input!
            return str(result)
        except Exception:
            return "Error: Could not perform calculation."
    elif tool_name == "schedule_event":
        title = parameters.get("title", "Event")
        date = parameters.get("date", "today")
        time = parameters.get("time", "now")
        attendees = ", ".join(parameters.get("attendees", []))
        return f"Event '{title}' scheduled for {date} at {time} with {attendees}." if attendees else f"Event '{title}' scheduled for {date} at {time}."
    else:
        return f"Unknown tool: {tool_name}"

# --- Interactive Inference Loop ---
def chat_loop():
    conversation_history = [] # Stores turns for context
    pformat_text("\nStart chatting! Type 'exit' to quit.", Color.BRIGHT_CYAN, Color.BOLD)

    while True:
        # Use format_text to correctly format the prompt string for input()
        user_input = input(format_text("\nUser: ", Color.YELLOW, Color.BOLD))
        if user_input.lower() == 'exit':
            break

        conversation_history.append({"role": "user", "content": user_input})
        prompt = format_inference_prompt(conversation_history)

        pformat_text(f"\n[DEBUG: Prompt sent to LLM]\n{prompt}\n[/DEBUG]", Color.DIM)

        # Generate response from the model
        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
        # Use reasonable generation parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, # Max length for the assistant's new response
            do_sample=True, # Enable sampling for more creative/diverse responses
            temperature=0.7, # Controls randomness (0.0 for deterministic, higher for more random)
            top_p=0.9, # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id, # Stop generation at EOS token
            pad_token_id=tokenizer.pad_token_id # Important for generation
        )
        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Clean up the generated text by removing the initial prompt tokens
        cleaned_response = generated_text.split("### Assistant:")[-1].strip()

        pformat_text(f"\n[DEBUG: LLM Raw Generated Response]\n{cleaned_response}\n[/DEBUG]", Color.DIM)

        # --- Tool Call Detection and Execution Simulation ---
        try:
            # Check if the generated response starts with a JSON object (indicating a tool call)
            if cleaned_response.startswith('{') and '"tool_name"' in cleaned_response:
                json_start = cleaned_response.find('{')
                json_end = cleaned_response.find('}', json_start) + 1
                tool_call_json_str = cleaned_response[json_start:json_end]

                tool_call_data = json.loads(tool_call_json_str)

                tool_name = tool_call_data.get("tool_name")
                parameters = tool_call_data.get("parameters", {})

                if tool_name and parameters is not None:
                    tool_output = execute_tool(tool_name, parameters)
                    pformat_text(f"Tool executed! Output: {tool_output}", Color.NORMAL_GREEN)

                    # Update the current turn in history with tool_call and tool_output
                    conversation_history[-1]["tool_call"] = tool_call_data
                    conversation_history[-1]["tool_output"] = tool_output

                    re_prompt = format_inference_prompt(conversation_history)
                    pformat_text(f"\n[DEBUG: Reprompt after tool execution]\n{re_prompt}\n[/DEBUG]", Color.DIM)

                    re_inputs = tokenizer(re_prompt, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
                    re_outputs = model.generate(
                        **re_inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    final_generated_text = tokenizer.decode(re_outputs[0][re_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    final_assistant_response = final_generated_text.split("### Assistant:")[-1].strip()

                    conversation_history[-1]["content"] = final_assistant_response
                    pformat_text(f"Assistant: {final_assistant_response}", Color.GREEN)

                else:
                    pformat_text(f"Assistant: {cleaned_response}", Color.NORMAL_WHITE)
                    conversation_history.append({"role": "assistant", "content": cleaned_response})
            else:
                pformat_text(f"Assistant: {cleaned_response}", Color.NORMAL_WHITE)
                conversation_history.append({"role": "assistant", "content": cleaned_response})

        except json.JSONDecodeError:
            pformat_text(f"Assistant: {cleaned_response}", Color.NORMAL_WHITE)
            conversation_history.append({"role": "assistant", "content": cleaned_response})
        except Exception as e:
            pformat_text(f"An error occurred during response processing: {e}", Color.RED)
            pformat_text(f"Assistant (raw): {cleaned_response}", Color.RED)
            conversation_history.append({"role": "assistant", "content": "I encountered an internal error."})


if __name__ == "__main__":
    chat_loop()
