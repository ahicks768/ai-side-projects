# AI Model Comparison Tool

Compares responses from Claude, GPT-4o, and Gemini across different system 
prompts, with Gemini acting as an automated judge to score outputs.

## Setup
1. Clone this repo
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys (see `.env.example`)
6. Run: `python3 read_score_article_update.py`

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

---

## Project 2: RAG-Powered Course Notes Q&A

Lets you ask questions about a set of PDF documents and get answers grounded in those documents rather than the model's general training knowledge. Built with LangChain, ChromaDB, and Streamlit.

## Setup
1. Clone this repo
2. Activate your virtual environment: `source ~/.venvs/ai-projects/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file with your Anthropic API key
5. Drop your PDFs into a `docs/` folder
6. Run: `streamlit run rag_web.py`

The first run builds and saves the vector store from your PDFs. After that it loads from disk and starts up quickly.

## Sample Output
User: Tell me about rules-based AI.

Answer: Based on the course notes, rules-based AI works by utilizing human judgment that is embedded in the form of rules about how to perform a task, and applying those rules to inputs in order to produce outputs. An example given in the notes is rules that specify how to recognize an image of George Washington. The notes also draw a contrast with supervised machine learning, which also uses human judgment, but delivers it differently through labeled data during the training process rather than explicitly coded rules. Beyond this, the notes do not go into further detail about the specific advantages or disadvantages of the rules-based approach.

Sources:
- Class 1 Introduction to AI and Machine Learning Technologies.pdf (page 2)
- Class 1 Introduction to AI and Machine Learning Technologies.pdf (page 9)

## What I Learned
Building this gave me a much clearer picture of how these systems actually work under the hood. A few things stood out.

The LLM is only one piece of the pipeline. The embedding model that converts text into vectors is a completely separate system, trained differently and for a different purpose. The vector store is another separate system on top of that. Something like LangChain just connects them. Understanding that distinction changed how I think about AI products generally, since most of them are really a combination of smaller components working together rather than one model doing everything.

RAG and fine-tuning are often talked about as if they are interchangeable, but they solve different problems. RAG retrieves information at query time without touching the model at all. Fine-tuning actually modifies the model's weights through additional training, and it is better at changing how a model behaves than teaching it new facts. For most use cases involving private or frequently updated data, RAG is the more practical choice.

The way retrieval works is also worth understanding. When you ask a question, it gets converted into the same type of vector as your documents and the app finds the closest match by measuring the angle between vectors, not by keyword search. That means it can find relevant content even when the exact words don't match, which is both useful and a potential point of failure if the embedding model doesn't represent certain concepts well.

## Known Limitations
The retriever always returns the top 4 chunks regardless of how relevant they actually are. Claude then decides from the text whether those chunks answer the question, so out-of-domain questions still trigger retrieval even if nothing useful comes back. There is also no memory across questions in the same session, and adding new documents requires rebuilding the vector store from scratch.