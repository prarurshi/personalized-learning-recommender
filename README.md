# Personalized Learning Resource Recommender (AI Lab Project)

A command-line **hybrid AI recommender system** that suggests courses, books, tutorials and projects
based on a learner's interests, goals, skill level, preferences and feedback.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`data/ratings.csv` is already included. To rebuild it: `python scripts/generate_ratings.py`.

## Features

1. **Learner profiles**: skill level, interests, goal, preferred formats, time limit and free-only option. Saved as JSON in `data/users/`.
2. **Personalized recommendations** with an **explanation** for each one ("Why recommended?") and a score breakdown.
3. **Keyword search**, ranked by TF-IDF relevance.
4. **Learning path generator**: builds an ordered path from the learner's level up to Advanced (Beginner → Intermediate → Advanced).
5. **Similar resources**: "more like this".
6. **Feedback loop**: rating a resource or marking it completed updates the model straight away.
7. **Offline evaluation**: Precision@K, Recall@K, Hit-rate and Coverage for every strategy, compared against a random baseline.

## AI techniques used

| Component | Technique | Library |
|---|---|---|
| Content-based filtering | TF-IDF vectorisation (unigrams + bigrams) and cosine similarity | scikit-learn |
| Relevance feedback | Rocchio algorithm: profile = interests + liked items − disliked items | numpy |
| Collaborative filtering | Item-item CF using adjusted-cosine similarity on the user-item rating matrix, with shrinkage | pandas, scikit-learn |
| Knowledge-based filtering | Rules for skill-level fit and preferred format; hard constraints for budget and duration | pandas |
| Hybridisation | Weighted hybrid of all signals; weights are re-normalised during cold start | numpy |
| Explainability | Rule-generated reasons for every recommendation | — |

```
final = 0.45·content + 0.25·collaborative + 0.20·knowledge + 0.10·popularity
```
You can change the weights in `recommender/config.py`. If a user has no ratings yet (cold start), the collaborative weight is shared out among the other signals.

## Sample evaluation (K = 5, 42 learners, 30 % of liked items hidden)

| strategy | precision@5 | recall@5 | hit_rate@5 |
|---|---|---|---|
| content | 0.205 | 0.536 | 0.714 |
| hybrid | 0.190 | 0.492 | 0.690 |
| collaborative | 0.067 | 0.149 | 0.333 |
| popular | 0.033 | 0.073 | 0.167 |
| random | 0.029 | 0.071 | 0.119 |

Hybrid and content-based filtering are about 6–7× better than random. The community ratings are synthetic and mostly driven by topic, which is why content-based filtering is so strong here.

## Project structure

```
learning_recommender/
├── main.py                  # entry point
├── requirements.txt
├── data/
│   ├── resources.csv        # 58 learning resources (catalogue)
│   ├── ratings.csv          # community ratings (synthetic)
│   └── users/               # saved learner profiles (JSON)
├── recommender/             # core AI logic - no print/input, reusable by any UI
│   ├── config.py            # paths, levels, formats, hybrid weights
│   ├── data_loader.py       # load/save CSV data
│   ├── profile.py           # UserProfile + ProfileStore
│   ├── engine.py            # RecommenderEngine (hybrid model, search, paths, similarity)
│   └── evaluation.py        # hold-out evaluation metrics
├── cli/
│   ├── app.py               # menu-driven CLI
│   └── display.py           # rich tables / panels
└── scripts/generate_ratings.py
```

## Next step: UI

The `recommender` package does not depend on the CLI. A UI (for example Streamlit) only needs to call the same methods:
`RecommenderEngine.recommend()`, `search()`, `learning_path()`, `similar_resources()`, `add_rating()` and `evaluate()`.

> Note: resource metadata (hours, ratings, cost) is approximate sample data for demonstration only.
