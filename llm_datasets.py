import json

# --- Test Dataset ---
test_dataset = [
  {
    "conversation_id": "test_001",
    "turns": [
      {"role": "user", "content": "What's the capital of Japan?"},
      {"role": "assistant",
       "tool_call": {"tool_name": "search_web", "parameters": {"query": "capital of Japan"}},
       "tool_output": "Search result: The capital and most populous city of Japan is Tokyo.",
       "content": "The capital of Japan is Tokyo."
      }
    ]
  },
  {
    "conversation_id": "test_002",
    "turns": [
      {"role": "user", "content": "Calculate 120 divided by 4."},
      {"role": "assistant",
       "tool_call": {"tool_name": "perform_calculation", "parameters": {"expression": "120 / 4"}},
       "tool_output": "30.0",
       "content": "120 divided by 4 is 30."
      }
    ]
  },
  {
    "conversation_id": "test_003",
    "turns": [
      {"role": "user", "content": "Schedule a meeting with Alex next Tuesday."},
      {"role": "assistant", "content": "What time should the meeting be?"},
      {"role": "user", "content": "At 2 PM."},
      {"role": "assistant",
       "tool_call": {"tool_name": "schedule_event", "parameters": {"title": "Meeting with Alex", "date": "next Tuesday", "time": "2:00 PM"}},
       "tool_output": "Event 'Meeting with Alex' scheduled for next Tuesday at 2:00 PM.",
       "content": "Okay, I've scheduled your meeting with Alex for next Tuesday at 2 PM."
      }
    ]
  },
  {
    "conversation_id": "test_004",
    "turns": [
      {"role": "user", "content": "Tell me something interesting."},
      {"role": "assistant", "content": "Did you know that a group of owls is called a parliament?"}
    ]
  },
  {
    "conversation_id": "test_005",
    "turns": [
      {"role": "user", "content": "What's the weather like in Berlin in Celsius?"},
      {"role": "assistant",
       "tool_call": {"tool_name": "get_weather", "parameters": {"location": "Berlin", "unit": "celsius"}},
       "tool_output": "Weather in Berlin: 22°C, sunny.",
       "content": "The weather in Berlin is 22°C and sunny."
      }
    ]
  }
]

# You can add your main training dataset here as well, e.g.:
training_dataset = [
    {
      "conversation_id": "train_dummy_001",
      "turns": [
        {"role": "user", "content": "What's the current time in Tokyo?"},
        {"role": "assistant",
         "tool_call": {"tool_name": "search_web", "parameters": {"query": "current time in Tokyo"}},
         "tool_output": "Search result: The current time in Tokyo is 11:30 AM.",
         "content": "The current time in Tokyo is 11:30 AM."
        }
      ]
    },
    {
      "conversation_id": "train_dummy_002",
      "turns": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
      ]
    },
    {
      "conversation_id": "train_dummy_003",
      "turns": [
        {"role": "user", "content": "Can you calculate 1+1?"},
        {"role": "assistant",
         "tool_call": {"tool_name": "perform_calculation", "parameters": {"expression": "1+1"}},
         "tool_output": "2",
         "content": "1 plus 1 is 2."
        }
      ]
    }
]
# For actual fine-tuning, `training_dataset` should be much larger and more diverse.
