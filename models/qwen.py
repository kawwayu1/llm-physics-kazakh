import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import InferenceClient
import config

client = InferenceClient(
    provider="together",
    api_key=config.HF_TOKEN,
)

def ask_qwen(problem_text):
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a physics teacher. Solve the given physics problem step by step. Show your reasoning clearly."
            },
            {
                "role": "user",
                "content": problem_text
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content