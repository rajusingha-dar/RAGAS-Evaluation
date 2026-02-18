"""
test_dataset.py
───────────────
Ground-truth Q&A pairs used by RAGAS for evaluation.

Each entry contains:
  - question       : the query to run through the pipeline
  - ground_truth   : the ideal answer (used for Context Recall and Answer Correctness)

Rules for writing good ground truth:
  1. Base every answer ONLY on what is in the documents — no outside knowledge
  2. Be specific — vague answers make Context Recall scoring unreliable
  3. Cover all three documents evenly
  4. Include a mix of: factual lookups, multi-detail questions, and definition questions
  5. Add 1-2 questions the documents cannot answer — to test "I don't have context" behaviour
"""

TEST_CASES = [

    # ── ai_basics.txt ─────────────────────────────────────────────────────────
    {
        "question": "What are the three types of machine learning?",
        "ground_truth": (
            "The three types of machine learning are supervised learning, "
            "unsupervised learning, and reinforcement learning."
        ),
    },
    {
        "question": "What is supervised learning and give an example of it?",
        "ground_truth": (
            "Supervised learning trains an algorithm on labeled data. "
            "An example is classifying emails as spam or not spam using historical labeled emails."
        ),
    },
    {
        "question": "What is deep learning and how does it relate to machine learning?",
        "ground_truth": (
            "Deep learning is a subset of machine learning that uses neural networks "
            "with many layers to model complex patterns in data."
        ),
    },
    {
        "question": "What neural network architecture is used for image recognition tasks?",
        "ground_truth": (
            "Convolutional Neural Networks (CNNs) are widely used for image tasks."
        ),
    },
    {
        "question": "What are the key ethical concerns around artificial intelligence?",
        "ground_truth": (
            "Key AI ethics concerns include algorithmic bias, privacy issues around data collection, "
            "lack of transparency in decision-making (the black box problem), "
            "and potential job displacement due to automation."
        ),
    },
    {
        "question": "What are large language models and what can they do?",
        "ground_truth": (
            "Large language models like GPT-4 are transformer-based models trained on vast amounts "
            "of text data. They can generate coherent text, answer questions, write code, "
            "and summarize documents."
        ),
    },

    # ── climate_change.txt ────────────────────────────────────────────────────
    {
        "question": "What is the primary cause of modern climate change?",
        "ground_truth": (
            "The primary cause of modern climate change is the increase of greenhouse gases "
            "in the atmosphere, particularly carbon dioxide released through burning fossil fuels."
        ),
    },
    {
        "question": "By how much have global average temperatures risen since pre-industrial times?",
        "ground_truth": (
            "Global average temperatures have risen by approximately 1.1 degrees Celsius "
            "since pre-industrial times."
        ),
    },
    {
        "question": "What is the greenhouse effect?",
        "ground_truth": (
            "The greenhouse effect works by trapping heat from the sun in Earth's atmosphere. "
            "Without it Earth would be too cold to support life, but excess emissions "
            "cause an enhanced greenhouse effect that is warming the planet."
        ),
    },
    {
        "question": "What does the Paris Agreement aim to achieve?",
        "ground_truth": (
            "The Paris Agreement, signed in 2015, committed nations to limiting warming "
            "to well below 2 degrees Celsius above pre-industrial levels, with efforts "
            "to limit it to 1.5 degrees Celsius."
        ),
    },
    {
        "question": "How much has the cost of solar panels dropped in the last decade?",
        "ground_truth": (
            "The cost of solar panels has dropped by over 90 percent in the last decade."
        ),
    },

    # ── space_exploration.txt ─────────────────────────────────────────────────
    {
        "question": "Who was the first human in space and when did it happen?",
        "ground_truth": (
            "Yuri Gagarin was the first human in space on April 12, 1961, "
            "completing one orbit of Earth aboard Vostok 1."
        ),
    },
    {
        "question": "When did Apollo 11 land on the Moon and who was first to walk on it?",
        "ground_truth": (
            "Apollo 11 landed on the Moon on July 20, 1969. "
            "Neil Armstrong was the first human to walk on the lunar surface."
        ),
    },
    {
        "question": "What is the International Space Station and who operates it?",
        "ground_truth": (
            "The International Space Station is a modular space station in low Earth orbit "
            "jointly operated by NASA, Roscosmos, ESA, JAXA, and CSA. "
            "It has been continuously inhabited since November 2000."
        ),
    },
    {
        "question": "What has SpaceX developed to reduce launch costs?",
        "ground_truth": (
            "SpaceX developed the Falcon 9 rocket with reusable first-stage boosters, "
            "dramatically reducing launch costs."
        ),
    },
    {
        "question": "What is NASA's Artemis program?",
        "ground_truth": (
            "NASA's Artemis program aims to return humans to the Moon by the mid-2020s, "
            "including the first woman and first person of color on the lunar surface, "
            "as a stepping stone to Mars."
        ),
    },

    # ── Out-of-context questions (should trigger 'I don't have enough context') ──
    {
        "question": "What is the current price of Bitcoin?",
        "ground_truth": (
            "The documents do not contain information about Bitcoin prices."
        ),
    },
    {
        "question": "Who won the FIFA World Cup in 2022?",
        "ground_truth": (
            "The documents do not contain information about the FIFA World Cup."
        ),
    },
]