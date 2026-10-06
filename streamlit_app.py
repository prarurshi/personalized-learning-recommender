"""Streamlit web UI for the Personalized Learning Resource Recommender.

Run:  streamlit run streamlit_app.py
Uses exactly the same `recommender` package as the CLI (main.py).
"""

import pandas as pd
import streamlit as st

from recommender import (FORMATS, LEVELS, STRATEGIES, ProfileStore, RecommenderEngine, UserProfile, evaluate,
                         load_ratings, load_resources, save_ratings)

st.set_page_config(page_title="Learning Resource Recommender", page_icon="▣", layout="wide")

LEVEL_CLASS = {"Beginner": "level-beginner", "Intermediate": "level-intermediate", "Advanced": "level-advanced"}


def inject_styles() -> None:
    """Create a small shadcn-inspired design system for Streamlit primitives."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --ink: #171717;
            --paper: #f4f0e8;
            --muted: #6f6a61;
            --line: #171717;
            --acid: #d8ff3e;
            --orange: #ff7048;
            --blue: #a8d8ff;
            --white: #fffdf8;
        }

        .stApp { background: var(--paper); color: var(--ink); }
        .stApp, .stApp p, .stApp label, .stApp button, .stApp input,
        .stApp textarea, .stApp select { font-family: 'Space Grotesk', sans-serif; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--ink); letter-spacing: -0.045em; }
        .stApp h1 { font-size: clamp(2.5rem, 5vw, 4.8rem); line-height: .95; }
        .stApp h2 { font-size: 2rem; }
        .stApp h3 { font-size: 1.35rem; }
        [data-testid='stHeader'] { background: transparent; }
        [data-testid='stSidebar'] { background: #e9e3d6; border-right: 2px solid var(--line); }
        [data-testid='stSidebar'] > div:first-child { padding-top: 1.5rem; }
        [data-testid='stSidebar'] h1 { font-size: 2.2rem; margin-bottom: .15rem; }

        .eyebrow, .utility-label, .section-kicker {
            font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .09em;
            font-size: .72rem; font-weight: 500;
        }
        .eyebrow { color: var(--orange); margin-bottom: .65rem; }
        .section-kicker { color: var(--muted); margin: 1.3rem 0 .25rem; }
        .utility-label { color: var(--muted); }
        .hero { padding: 1.1rem 0 1.7rem; border-bottom: 2px solid var(--line); margin-bottom: 1.25rem; }
        .hero p { color: var(--muted); font-size: 1.05rem; max-width: 760px; }
        .hero-mark { display: inline-block; background: var(--acid); border: 2px solid var(--line); padding: .25rem .45rem; transform: rotate(-2deg); }
        .brutal-note { background: var(--orange); border: 2px solid var(--line); box-shadow: 5px 5px 0 var(--line); padding: 1rem 1.1rem; }
        .brutal-note strong { font-size: 1.1rem; }
        .stat-card { background: var(--white); border: 2px solid var(--line); box-shadow: 4px 4px 0 var(--line); padding: .9rem 1rem; min-height: 92px; }
        .stat-value { font-size: 1.7rem; font-weight: 700; line-height: 1; margin-top: .35rem; }
        .stat-label { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }
        .resource-card { background: var(--white); border: 2px solid var(--line); box-shadow: 4px 4px 0 var(--line); padding: .82rem 1rem; margin: .22rem 0 .58rem; }
        .resource-card h4 { margin: 0 0 .38rem; font-size: 1.2rem; }
        .resource-meta { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .73rem; line-height: 1.7; }
        .resource-description { color: #4f4a43; font-size: .9rem; line-height: 1.45; margin: .65rem 0 0; }
        .tag { display: inline-block; border: 1.5px solid var(--line); padding: .12rem .38rem; margin-right: .25rem; font-family: 'DM Mono', monospace; font-size: .68rem; background: var(--blue); }
        .level-badge, .status-badge { display: inline-flex; align-items: center; gap: .3rem; border: 1.5px solid var(--line); padding: .12rem .4rem; font-family: 'DM Mono', monospace; font-size: .66rem; font-weight: 700; text-transform: uppercase; }
        .level-badge::before { content: ''; width: .42rem; height: .42rem; border: 1px solid var(--line); background: currentColor; display: inline-block; }
        .level-beginner { color: #237a3b; background: #d8f4dc; }
        .level-intermediate { color: #8a5b00; background: #fff0b8; }
        .level-advanced { color: #a52d24; background: #ffd7d2; }
        .level-default { color: var(--ink); background: var(--blue); }
        .status-badge { color: #176b39; background: #d8f4dc; }
        .recommendation-list > div[data-testid='stVerticalBlock'] { gap: .1rem; }
        .stButton > button, .stFormSubmitButton > button { border: 2px solid var(--line); border-radius: 0; box-shadow: 3px 3px 0 var(--line); font-weight: 700; transition: transform .12s, box-shadow .12s; }
        .stButton > button:hover, .stFormSubmitButton > button:hover { transform: translate(2px, 2px); box-shadow: 1px 1px 0 var(--line); }
        .stButton > button[kind='primary'], .stFormSubmitButton > button[kind='primary'] { background: var(--acid); color: var(--ink); }
        div[data-baseweb='input'] > div, div[data-baseweb='select'] > div, textarea, .stTextInput input,
        .stNumberInput input { border: 2px solid var(--line) !important; border-radius: 0 !important; background: var(--white) !important; }
        div[data-testid='stTabs'] button { font-weight: 700; color: var(--muted); }
        div[data-testid='stTabs'] button[aria-selected='true'] { color: var(--ink); border-bottom-color: var(--orange); }
        .stMetric { background: var(--white); border: 2px solid var(--line); padding: .75rem; box-shadow: 3px 3px 0 var(--line); }
        .stDataFrame { border: 2px solid var(--line); }
        .stAlert { border: 2px solid var(--line); border-radius: 0; box-shadow: 3px 3px 0 var(--line); }
        hr { border-color: var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, tone: str = "") -> None:
    st.markdown(
        f"<div class='stat-card {tone}'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def level_badge(level: str) -> str:
    level_class = LEVEL_CLASS.get(level, "level-default")
    return f"<span class='level-badge {level_class}'>{level}</span>"


inject_styles()

# ------------------------------------------------------------------ app state
if "engine" not in st.session_state:
    st.session_state.engine = RecommenderEngine(load_resources(), load_ratings())
    st.session_state.store = ProfileStore()
    st.session_state.profile = None

engine: RecommenderEngine = st.session_state.engine
store: ProfileStore = st.session_state.store
TOPICS = sorted(engine.resources["topic"].unique())


# ------------------------------------------------------------------ callbacks
def rate_cb(rid: int, key: str) -> None:
    value = st.session_state.get(key)
    if value is None:
        return
    rating = value + 1  # st.feedback("stars") returns 0..4
    p = st.session_state.profile
    p.rate(rid, rating)
    engine.add_rating(p.username, rid, rating)  # model learns immediately
    save_ratings(engine.ratings)
    store.save(p)
    st.toast(f"Saved rating {rating}/5 — your recommendations have been updated!")


def complete_cb(rid: int) -> None:
    p = st.session_state.profile
    p.mark_completed(rid)
    store.save(p)
    st.toast("Marked as completed")


# ------------------------------------------------------------------ components
def resource_card(r: dict, prefix: str, score=None, score_label="Match", reasons=None, breakdown=None):
    p = st.session_state.profile
    rid = int(r["id"])
    with st.container():
        left, right = st.columns([5, 2])
        with left:
            cost = "FREE" if r["cost"].lower() == "free" else "PAID"
            st.markdown(
                f"<div class='resource-card'><div class='resource-meta'>RESOURCE / {int(r['id']):03d}</div>"
                f"<h4>{r['title']}</h4>"
                f"<div class='resource-meta'>{level_badge(r['level'])} "
                f"<span class='tag'>{r['topic'].upper()}</span> {r['format']} · {r['duration_hours']:g} H · "
                f"{cost} · ★ {r['rating']} · {r['provider']}</div>"
                f"<div class='resource-description'>{r['description']}</div></div>",
                unsafe_allow_html=True,
            )
            if reasons:
                st.markdown("**WHY THIS MATCHES** · " + " · ".join(reasons))
            if breakdown:
                st.caption("SIGNAL BREAKDOWN · " + ", ".join(
                    f"{k}: {v:.2f}" for k, v in breakdown.items() if v is not None))
        with right:
            if score is not None:
                st.metric(score_label, f"{score:.0%}" if score_label == "Match" else f"{score:.2f}")
            if p:
                current = p.ratings.get(rid)
                st.caption(f"Your rating: {'★' * int(current)}" if current else "Rate this resource:")
                key = f"{prefix}_rate_{rid}"
                st.feedback("stars", key=key, on_change=rate_cb, args=(rid, key))
                if rid in p.completed:
                    st.markdown("<span class='status-badge'>Completed</span>", unsafe_allow_html=True)
                else:
                    st.button("Mark completed", key=f"{prefix}_done_{rid}", on_click=complete_cb, args=(rid,))


def preferences_form(p, form_key: str) -> None:
    """Create a new profile (p is None) or edit an existing one."""
    creating = p is None
    with st.form(form_key):
        username = st.text_input("Username *", help="3-30 letters, digits, _ or -") if creating else p.username
        name = st.text_input("Name", value="" if creating else p.name)
        level = st.selectbox("Skill level", LEVELS, index=0 if creating else LEVELS.index(p.level))
        topics = st.multiselect("Topics of interest", TOPICS,
                                default=[] if creating else [i for i in p.interests if i in TOPICS])
        extra = st.text_input("Extra keywords (comma separated)", placeholder="pytorch, nlp, sql",
                              value="" if creating else ", ".join(i for i in p.interests if i not in TOPICS))
        goals = st.text_area("Learning goal", value="" if creating else p.goals,
                             placeholder="e.g. become a machine learning engineer")
        formats = st.multiselect("Preferred formats (empty = any)", FORMATS,
                                 default=[] if creating else p.preferred_formats)
        max_h = st.number_input("Max hours per resource (0 = no limit)", min_value=0, max_value=500, step=5,
                                value=0 if creating else int(p.max_duration or 0))
        free = st.checkbox("Free resources only", value=False if creating else p.free_only)
        submitted = st.form_submit_button("Create profile" if creating else "Save changes", type="primary")

    if not submitted:
        return
    if creating:
        username = username.strip().lower()
        if not store.valid_username(username):
            st.error("Username must be 3-30 characters: letters, digits, _ or -")
            return
        if store.exists(username):
            st.error("That username already exists - please log in instead.")
            return
        p = UserProfile(username=username)
    p.name = name.strip()
    p.level = level
    p.interests = topics + [k.strip() for k in extra.split(",") if k.strip()]
    p.goals = goals.strip()
    p.preferred_formats = formats
    p.max_duration = float(max_h) if max_h > 0 else None
    p.free_only = free
    store.save(p)
    st.session_state.profile = p
    st.rerun()


@st.cache_data(show_spinner=False)
def run_evaluation(ratings: pd.DataFrame, k: int) -> pd.DataFrame:
    return evaluate(engine.resources, ratings, k=k)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("<div class='eyebrow'>PERSONALIZED LEARNING / 01</div>", unsafe_allow_html=True)
    st.title("Learner")
    st.markdown("<div class='utility-label'>YOUR CURRICULUM, RECOMPOSED.</div>", unsafe_allow_html=True)
    profile = st.session_state.profile
    if profile is None:
        mode = st.radio("Account", ["Log in", "Create profile"], horizontal=True, label_visibility="collapsed")
        if mode == "Log in":
            users = store.list_users()
            if users:
                chosen = st.selectbox("Select your profile", users)
                if st.button("Log in", type="primary", width="stretch"):
                    st.session_state.profile = store.load(chosen)
                    st.rerun()
            else:
                st.info("No profiles yet - switch to **Create profile**.")
        else:
            preferences_form(None, "create_form")
    else:
        st.markdown(f"### Hi, {profile.name or profile.username}")
        st.markdown(f"{level_badge(profile.level)} <span class='utility-label'>{len(profile.ratings)} RATED · "
                    f"{len(profile.completed)} COMPLETED</span>", unsafe_allow_html=True)
        st.caption("Interests: " + (", ".join(profile.interests) or "-"))
        with st.expander("Edit preferences"):
            preferences_form(profile, "edit_form")
        if st.button("Log out", width="stretch"):
            st.session_state.profile = None
            st.rerun()
    st.divider()
    st.markdown(f"<div class='utility-label'>{len(engine.resources)} RESOURCES &nbsp;·&nbsp; {len(engine.ratings)} COMMUNITY RATINGS</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ main page
st.markdown(
    "<div class='hero'><div class='eyebrow'>LEARNING OS / RECOMMENDATION ENGINE</div>"
    "<h1>Build a <span class='hero-mark'>better</span> learning loop.</h1>"
    "<p>Personalized resources ranked by what you want to learn, how you learn best, and what the community found useful.</p></div>",
    unsafe_allow_html=True,
)

profile = st.session_state.profile
if profile is None:
    st.markdown("<div class='brutal-note'><strong>Start with your signal.</strong><br>Create a profile or log in from the sidebar to unlock personalized recommendations.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>COMMUNITY FAVOURITES</div><h2>Top-rated resources</h2>", unsafe_allow_html=True)
    for r in engine.resources.sort_values("rating", ascending=False).head(6).to_dict("records"):
        resource_card(r, "popular")
    st.stop()

tab_rec, tab_search, tab_path, tab_browse, tab_profile, tab_eval = st.tabs(
    ["Recommendations", "Search", "Learning Path", "Browse & Similar", "My Profile", "Evaluation"])

# ---- Recommendations
with tab_rec:
    st.markdown("<div class='section-kicker'>YOUR FEED / RANKED BY FIT</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    n = c1.slider("How many?", 3, 20, 5)
    strategy = c2.selectbox("Strategy", STRATEGIES, format_func=str.title,
                            help="Hybrid combines all signals. Others are shown for comparison.")
    topic = c3.selectbox("Restrict to topic", ["Any"] + TOPICS)
    if not profile.ratings:
        st.info("Rate a few resources with the star control — the collaborative model will start using your taste.")
    recs = engine.recommend(profile, top_n=n, strategy=strategy, topic=None if topic == "Any" else topic)
    if not recs:
        st.warning("No resources match your constraints. Try relaxing max hours / free-only in your preferences.")
    for rec in recs:
        resource_card(rec.resource, "rec", rec.score, reasons=rec.reasons, breakdown=rec.breakdown)

# ---- Search
with tab_search:
    query = st.text_input("Search by keywords", placeholder="e.g. neural networks pytorch")
    if query:
        results = engine.search(query, top_n=10)
        if not results:
            st.warning("No results found.")
        for r, s in results:
            resource_card(r, "search", s, score_label="Relevance")

# ---- Learning path
with tab_path:
    c1, c2 = st.columns([3, 1])
    goal = c1.text_input("What do you want to learn?", value=profile.goals or "",
                         placeholder="e.g. deep learning, web development with python")
    per_level = c2.slider("Resources per level", 1, 4, 2)
    if goal:
        stages = engine.learning_path(profile, goal, per_level=per_level)
        if not stages:
            st.warning("No matching resources found for that goal.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Stages", len(stages))
            m2.metric("Resources", sum(len(s["resources"]) for s in stages))
            m3.metric("Total time", f"{sum(s['hours'] for s in stages):g} h")
            step = 1
            for stage in stages:
                st.markdown(f"<h3>{level_badge(stage['level'])} {stage['level']} stage · {stage['hours']:g} h</h3>", unsafe_allow_html=True)
                for r in stage["resources"]:
                    st.markdown(f"**Step {step}**")
                    resource_card(r, "path")
                    step += 1

# ---- Browse & similar
with tab_browse:
    df = engine.resources
    c1, c2, c3, c4 = st.columns(4)
    f_topics = c1.multiselect("Topic", TOPICS)
    f_levels = c2.multiselect("Level", LEVELS)
    f_formats = c3.multiselect("Format", FORMATS)
    f_free = c4.checkbox("Free only")
    if f_topics:
        df = df[df["topic"].isin(f_topics)]
    if f_levels:
        df = df[df["level"].isin(f_levels)]
    if f_formats:
        df = df[df["format"].isin(f_formats)]
    if f_free:
        df = df[df["cost"] == "Free"]
    st.dataframe(df[["id", "title", "topic", "level", "format", "duration_hours", "cost", "rating", "provider"]],
                 hide_index=True, width="stretch",
                 column_config={"duration_hours": st.column_config.NumberColumn("Hours"),
                                "rating": st.column_config.NumberColumn("Rating", format="%.1f")})
    st.subheader("Find similar resources")
    titles = dict(zip(engine.resources["title"], engine.resources["id"]))
    base = st.selectbox("Pick a resource you liked", list(titles), index=None, placeholder="Choose a resource")
    if base:
        for r, s in engine.similar_resources(titles[base], top_n=5):
            resource_card(r, "similar", s, score_label="Similarity")

# ---- Profile
with tab_profile:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Level", profile.level)
    m2.metric("Rated", len(profile.ratings))
    m3.metric("Completed", len(profile.completed))
    avg = sum(profile.ratings.values()) / len(profile.ratings) if profile.ratings else 0
    m4.metric("Avg rating given", f"{avg:.1f} ★" if profile.ratings else "-")
    ids = sorted(set(profile.ratings) | set(profile.completed))
    if not ids:
        st.info("No history yet. Rate or complete resources to build your learning history.")
    else:
        hist = pd.DataFrame([{
            "id": i, "title": engine.get_resource(i)["title"], "topic": engine.get_resource(i)["topic"],
            "your rating": "★" * int(profile.ratings[i]) if i in profile.ratings else "-",
            "completed": "Completed" if i in profile.completed else "",
        } for i in ids if engine.get_resource(i)])
        st.dataframe(hist, hide_index=True, width="stretch")
        st.markdown("**Your activity by topic**")
        st.bar_chart(hist["topic"].value_counts())

# ---- Evaluation
with tab_eval:
    st.markdown("Hold-out evaluation: for each community learner, 30% of their liked resources are hidden, "
                "then each strategy tries to recover them in its top-K list. **Random** is the baseline.")
    k = st.slider("K (list length)", 3, 15, 5)
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            st.session_state.eval = (k, run_evaluation(engine.ratings, k))
    if "eval" in st.session_state:
        k_used, res = st.session_state.eval
        st.caption(f"{res.attrs.get('users_evaluated', '?')} learners evaluated, K = {k_used}")
        st.dataframe(res, hide_index=True, width="stretch")
        st.bar_chart(res.set_index("strategy")[[f"precision@{k_used}", f"recall@{k_used}", f"hit_rate@{k_used}"]],
                     stack=False)
