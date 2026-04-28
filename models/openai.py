import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def ask_gpt(problem_text):
    response = client.chat.completions.create(
        model="gpt-4o",
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