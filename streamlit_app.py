"""Streamlit web UI for the Personalized Learning Resource Recommender.

Run:  streamlit run streamlit_app.py
Uses exactly the same `recommender` package as the CLI (main.py).
"""

import pandas as pd
import streamlit as st

from recommender import (FORMATS, LEVELS, STRATEGIES, ProfileStore, RecommenderEngine, UserProfile, evaluate,
                         load_ratings, load_resources, save_ratings)

st.set_page_config(page_title="Learning Resource Recommender", page_icon="🎓", layout="wide")

LEVEL_ICON = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}

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
    st.toast(f"Saved {rating}★ - your recommendations have been updated!", icon="✅")


def complete_cb(rid: int) -> None:
    p = st.session_state.profile
    p.mark_completed(rid)
    store.save(p)
    st.toast("Marked as completed", icon="🎉")


# ------------------------------------------------------------------ components
def resource_card(r: dict, prefix: str, score=None, score_label="Match", reasons=None, breakdown=None):
    p = st.session_state.profile
    rid = int(r["id"])
    with st.container(border=True):
        left, right = st.columns([5, 2])
        with left:
            st.markdown(f"##### {r['title']}")
            cost = "🆓 Free" if r["cost"].lower() == "free" else "💲 Paid"
            st.markdown(f"{LEVEL_ICON.get(r['level'], '')} **{r['level']}** · {r['topic']} · 📘 {r['format']} · "
                        f"⏱️ {r['duration_hours']:g} h · {cost} · ⭐ {r['rating']} · *{r['provider']}*")
            st.caption(r["description"])
            if reasons:
                st.markdown("**Why recommended:** " + " · ".join(reasons))
            if breakdown:
                st.caption("Score breakdown - " + ", ".join(
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
                    st.success("Completed", icon="✅")
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
    st.title("🎓 Learner")
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
        st.markdown(f"### 👋 Hi, {profile.name or profile.username}")
        st.caption(f"{LEVEL_ICON[profile.level]} {profile.level} · {len(profile.ratings)} rated · "
                   f"{len(profile.completed)} completed")
        st.caption("Interests: " + (", ".join(profile.interests) or "-"))
        with st.expander("✏️ Edit preferences"):
            preferences_form(profile, "edit_form")
        if st.button("Log out", width="stretch"):
            st.session_state.profile = None
            st.rerun()
    st.divider()
    st.caption(f"📚 {len(engine.resources)} resources · 👥 {len(engine.ratings)} community ratings")

# ------------------------------------------------------------------ main page
st.title("🎓 Personalized Learning Resource Recommender")
st.caption("Hybrid AI recommender: TF-IDF content-based + item-item collaborative + knowledge-based filtering")

profile = st.session_state.profile
if profile is None:
    st.info("👈 **Create a profile** or **log in** from the sidebar to get personalized recommendations.")
    st.subheader("🔥 Top-rated resources")
    for r in engine.resources.sort_values("rating", ascending=False).head(6).to_dict("records"):
        resource_card(r, "popular")
    st.stop()

tab_rec, tab_search, tab_path, tab_browse, tab_profile, tab_eval = st.tabs(
    ["🎯 Recommendations", "🔍 Search", "🗺️ Learning Path", "📚 Browse & Similar", "👤 My Profile", "📊 Evaluation"])

# ---- Recommendations
with tab_rec:
    c1, c2, c3 = st.columns(3)
    n = c1.slider("How many?", 3, 20, 5)
    strategy = c2.selectbox("Strategy", STRATEGIES, format_func=str.title,
                            help="Hybrid combines all signals. Others are shown for comparison.")
    topic = c3.selectbox("Restrict to topic", ["Any"] + TOPICS)
    if not profile.ratings:
        st.info("💡 Rate a few resources with the ⭐ stars - the collaborative model will start using your taste.")
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
                st.subheader(f"{LEVEL_ICON[stage['level']]} {stage['level']} stage · {stage['hours']:g} h")
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
                                "rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐")})
    st.subheader("🔗 Find similar resources")
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
            "completed": "✅" if i in profile.completed else "",
        } for i in ids if engine.get_resource(i)])
        st.dataframe(hist, hide_index=True, width="stretch")
        st.markdown("**Your activity by topic**")
        st.bar_chart(hist["topic"].value_counts())

# ---- Evaluation
with tab_eval:
    st.markdown("Hold-out evaluation: for each community learner, 30% of their liked resources are hidden, "
                "then each strategy tries to recover them in its top-K list. **Random** is the baseline.")
    k = st.slider("K (list length)", 3, 15, 5)
    if st.button("▶ Run evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            st.session_state.eval = (k, run_evaluation(engine.ratings, k))
    if "eval" in st.session_state:
        k_used, res = st.session_state.eval
        st.caption(f"{res.attrs.get('users_evaluated', '?')} learners evaluated, K = {k_used}")
        st.dataframe(res, hide_index=True, width="stretch")
        st.bar_chart(res.set_index("strategy")[[f"precision@{k_used}", f"recall@{k_used}", f"hit_rate@{k_used}"]],
                     stack=False)
