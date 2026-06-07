import anthropic
import openai
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

prompt = "Hello, who are you?"

# --- Claude ---
claude_response = claude_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}]
)
print("=== Claude ===")
print(claude_response.content[0].text)
print()

# --- OpenAI ---
openai_response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)
print("=== OpenAI ===")
print(openai_response.choices[0].message.content)
print()

# --- Gemini ---
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
gemini_response = gemini_model.generate_content(prompt)
print("=== Gemini ===")
print(gemini_response.text)