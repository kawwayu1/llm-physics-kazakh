import pandas as pd
from openai import OpenAI
import anthropic
from huggingface_hub import InferenceClient
import time
import config

# clients
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
hf_client = InferenceClient(provider="together", api_key=config.HF_TOKEN)

# data loading process
df_problems = pd.read_excel('data/problems.xlsx')
df_results = pd.read_excel('data/results.xlsx')
df_results['Model_Answer'] = df_results['Model_Answer'].astype(str)

def ask_gpt(text):
    r = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a physics teacher. Solve the given physics problem step by step. Write all math using plain text, not LaTeX."},
            {"role": "user", "content": text}
        ],
        max_tokens=1000
    )
    return r.choices[0].message.content

def ask_claude(text):
    r = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system="You are a physics teacher. Solve the given physics problem step by step. Write all math using plain text, not LaTeX.",
        messages=[{"role": "user", "content": text}]
    )
    return r.content[0].text

def ask_qwen(text):
    r = hf_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a physics teacher. Solve the given physics problem step by step. Write all math using plain text, not LaTeX."},
            {"role": "user", "content": text}
        ],
        max_tokens=1000
    )
    return r.choices[0].message.content

def run_experiment():
    for _, row in df_problems.iterrows():
        problem_id = row['Problem_ID']
        for language in ['English', 'Kazakh']:
            problem_text = row['Eng_version'] if language == 'English' else row['Kaz_version']

            if pd.isna(problem_text) or str(problem_text).strip() == '':
                continue

            print(f"\nProblem {problem_id} | {language}")

            for model_name, func in [('GPT-4o', ask_gpt), ('Claude Sonnet', ask_claude), ('Qwen2.5-7B', ask_qwen)]:
                try:
                    answer = func(problem_text)
                    print(f"  {model_name}: done")
                except Exception as e:
                    answer = f"ERROR: {e}"
                    print(f"  {model_name}: error - {e}")

                mask = (
                    (df_results['Problem_ID'] == problem_id) &
                    (df_results['Language'] == language) &
                    (df_results['Model'] == model_name)
                )
                df_results.loc[mask, 'Model_Answer'] = answer
                time.sleep(1)

        df_results.to_excel('data/results.xlsx', index=False)
        print(f"Saved problem {problem_id}")

if __name__ == "__main__":
    run_experiment()