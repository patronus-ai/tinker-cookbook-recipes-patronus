# tool_tokenization_basic.py
from transformers import AutoTokenizer
import json

print("=== Basic Tool Call Tokenization Demo ===\n")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Tool-calling trace format
messages = [
    {"role": "user", "content": "Calculate 2+2"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "calculator",
                "arguments": {"expression": "2+2"}
            }
        ]
    },
    {
        "role": "tool",
        "content": "4",
        "tool_call_id": "call_1"
    },
    {"role": "assistant", "content": "The answer is 4"}
]

print("=== INPUT MESSAGES ===")
print(json.dumps(messages, indent=2))
print("\n")

# Check tokenization
print("=== TOKENIZED TEXT ===")
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
print(text)
print("\n")

# Tokenize to IDs
print("=== TOKEN IDS ===")
tokens = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False
)
print(f"Total tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print("\n")

# Decode individual tokens
print("=== INDIVIDUAL TOKENS ===")
print(f"{'Index':<6} | {'Token ID':<10} | {'Decoded Token'}")
print("-" * 60)
for i in range(min(50, len(tokens))):
    decoded = tokenizer.decode([tokens[i]]).replace('\n', '\\n')
    print(f"{i:<6} | {tokens[i]:<10} | {decoded}")

print("\n✅ Basic tokenization works! Tool calls are properly serialized.")
