import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anthropic
import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

def ask_claude(problem_text):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": problem_text
            }
        ],
        system="You are a physics teacher. Solve the given physics problem step by step. Show your reasoning clearly."
    )
    return response.content[0].text