import anthropic
import openai
import os
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

article = """CHICAGO — The U.S. soccer team's final tuneup before the World Cup — a Saturday send-off against four-time champion Germany at sunny, sold-out Soldier Field — was part party, part pep rally but mostly one last semi-serious chance for Mauricio Pochettino to prepare his team for its moment in the summer spotlight.

The festive atmosphere belied the gravity of what lies ahead a half-continent away in six days when Pochettino's 19 months of curating and shaping, cajoling and adjusting, tweaking and testing will face judgement.

The 2-1 defeat before 63,636 spectators featured defensive frailty and Antonee Robinson's thunderous tying goal late in the first half. It brought spells of quality play against a quality opponent but also worrying signs about the Americans' capability of making a deep run when teams such as Germany stand in the way.

After the match, the Americans were scheduled to fly to Southern California and arrive at base camp ahead of the Group D opener against Paraguay on Friday night at SoFi Stadium.

They will also face Australia on June 19 in Seattle and Türkiye on June 25 back in Inglewood, California. In a city that did not submit a World Cup bid because of FIFA's contract terms, fans arriving early at Chicago's lakefront venue created a World Cup-worthy atmosphere with their display of colors and sounds. With deep heritage in the region, the Germans enjoyed a good swath of the sellout support.

There was a pregame flashback to the last time the World Cup was held in this country, with 15 members of the 1994 U.S. squad, including John Harkes and Cobi Jones, honored on the field.

All 26 current U.S. players, not just the starters, were introduced before kickoff. The group included center back Chris Richards, who wasn't available again because of an ankle injury. It's unclear whether he is on pace to play in the opener. In a worst-case scenario, a permanent roster change would have to be made 24 hours before the opener. After mixing probable starters and reserves in the 3-2 victory over Senegal last Sunday in Charlotte, Pochettino selected a lineup that seemed close to his World Cup picks, sans Richards, whose slot was filled by Miles Robinson.

Matt Freese returned to goal and Antonee Robinson and Sergiño Dest retained their places as wing backs or fullbacks, depending on the situation. Tim Ream and Alex Freeman, who started against Senegal, joined Miles Robinson on the backline.

Tyler Adams and Malik Tillman were in central midfield supporting Christian Pulisic, Weston McKennie and striker Folarin Balogun.

The fun backdrop and inconsequential outcome could not shroud the harsh realities facing the U.S. defense entering the World Cup. Less than two minutes after the opening whistle, the Germans scored on a set piece as if they were playing against an under-15 team. (All due respect to U-15s everywhere.)

Captain Joshua Kimmich served a free kick from maybe 30 yards. A week removed from scoring for Arsenal in the Champions League final, Kai Havertz gained inside position on Miles Robinson after Ream had left him.

Freese remained rooted to his line, all but helpless to what was to come.

As he reached the 6-yard box, Havertz headed it home without a challenge. He began his restrained celebration by holding up his palms, as if to say, "I could've done that in my sleep."""

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