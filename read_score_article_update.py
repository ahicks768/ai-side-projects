import anthropic
import openai
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

load_dotenv()

claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt_1 = "Summarize the following article in 3 sentences."
prompt_2 = "You are an editor at a major newspaper. Summarize the following article for a busy executive who has 30 seconds to read it."
prompt_3 = "Summarize the following article. Output exactly 3 bullet points: the main event, why it matters, and what happens next."

prompts = [prompt_1, prompt_2, prompt_3]

url = input("Paste article URL: ").strip()
page = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
page.raise_for_status()
soup = BeautifulSoup(page.text, "html.parser")
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()
article = "\n\n".join(p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40)

# Store responses for judging
claude_responses = []
openai_responses = []

# Claude loop
for i, system_prompt in enumerate(prompts, start=1):
    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": article}]
    )
    text = response.content[0].text
    claude_responses.append(text)
    print(f"=== Claude — Prompt {i} ===")
    print(text)
    print()

# OpenAI loop
for i, system_prompt in enumerate(prompts, start=1):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article}
        ]
    )
    text = response.choices[0].message.content
    openai_responses.append(text)
    print(f"=== OpenAI — Prompt {i} ===")
    print(text)
    print()

# Build judging prompt
judging_prompt = """You are an impartial evaluator. Below are 6 summaries of the same article, produced by two AI models (Claude and ChatGPT) each responding to 3 different system prompts.

For each prompt, compare the two responses and score each on:
- Conciseness (1-5): Is it appropriately brief without losing key information?
- Format-following (1-5): Did it follow the system prompt's instructions?
- Readability (1-5): Is it clear and easy to read?

Then pick a winner for each prompt.

SYSTEM PROMPT 1: "{p1}"
Claude: {c1}
ChatGPT: {o1}

SYSTEM PROMPT 2: "{p2}"
Claude: {c2}
ChatGPT: {o2}

SYSTEM PROMPT 3: "{p3}"
Claude: {c3}
ChatGPT: {o3}

Provide scores and a 1-2 sentence explanation for each prompt, then an overall winner.
""".format(
    p1=prompts[0], c1=claude_responses[0], o1=openai_responses[0],
    p2=prompts[1], c2=claude_responses[1], o2=openai_responses[1],
    p3=prompts[2], c3=claude_responses[2], o3=openai_responses[2]
)

# Gemini judges
print("=== Gemini's Evaluation ===")
gemini_response = gemini_client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=judging_prompt
)
print(gemini_response.text)