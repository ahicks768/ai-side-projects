# AI Model Comparison Tool

Compares responses from Claude, GPT-4o, and Gemini across different system 
prompts, with Gemini acting as an automated judge to score outputs.

## Setup
1. Clone this repo
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys (see `.env.example`)
6. Run: `python3 read_and_score_article.py`

## Sample Output
=== Claude — Prompt 3 ===
    • Main Event: The U.S. men's national soccer team lost 2-1 to Germany in a World Cup send-off friendly at Soldier Field in Chicago, featuring a defensive breakdown leading to an early German goal and a highlight-reel equalizer from Antonee Robinson before halftime.

    • Why It Matters: The result exposed concerning defensive vulnerabilities — particularly on set pieces — just days before the U.S. opens World Cup Group D play against Paraguay, raising questions about the team's ability to advance deep into the tournament against top competition.

    • What Happens Next: The team flies to Southern California to set up base camp, with their World Cup campaign beginning Friday night against Paraguay at SoFi Stadium, followed by matches against Australia on June 19 in Seattle and Türkiye on June 25 in Inglewood; the fitness status of injured center back Chris Richards remains uncertain ahead of the opener.

=== OpenAI — Prompt 1 ===
    The U.S. soccer team's final pre-World Cup match resulted in a 2-1 defeat against Germany at Soldier Field, highlighting some defensive frailties despite showing promising moments of quality play. With the World Cup fast approaching, coach Mauricio Pochettino's squad must address these vulnerabilities as they prepare for their Group D opener against Paraguay in Southern California. As part of their send-off, the event at Chicago featured both a festive atmosphere with celebrations and references to former U.S. players, while also serving as a critical evaluation of the team's readiness for the upcoming international challenges.

## What I Learned
One of the most important things that I recognized running this is that these models are not necessarily reliable evaluators of quality.  There may be some degree of objectiveness to the scoring algorithms (concision), but ultimately the scores are heavily subjective and a good score may just mean that the models are more similar in their training.  This may seem obvious as any evaluation (even from a human) has bias, but it is worth noting here as it can be easy to start to trust AI and overrely on it as it improves.

## Known Limitations
First, the user of the program has to manually input the code from the article that they want reviewed.  Another limitation is only comparing two frontier models right now.  One more serious limitation here is the lack of criteria for what defines a strong grade.  One AI model's understanding of a good article may be different from another's, making the scoring quite subjective.