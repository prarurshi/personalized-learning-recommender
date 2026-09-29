"""Interactive menu-driven command line application."""

from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from recommender import (FORMATS, LEVELS, STRATEGIES, ProfileStore, RecommenderEngine, UserProfile, evaluate,
                         load_ratings, load_resources, save_ratings)

from . import display as ui
from .display import console


# ---------------------------------------------------------------- input helpers
def choose_one(prompt: str, options: list, default: int = 1) -> str:
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}[/]. {opt}")
    idx = IntPrompt.ask(prompt, choices=[str(i) for i in range(1, len(options) + 1)],
                        default=default, show_choices=False)
    return options[idx - 1]


def choose_many(prompt: str, options: list, current: list = None) -> list:
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}[/]. {opt}")
    hint = f" (current: {', '.join(current)})" if current else ""
    while True:
        raw = Prompt.ask(f"{prompt} - numbers separated by commas, blank for none{hint}", default="")
        if not raw.strip():
            return []
        try:
            picks = [int(x) for x in raw.replace(" ", "").split(",") if x]
            if all(1 <= p <= len(options) for p in picks):
                return list(dict.fromkeys(options[p - 1] for p in picks))
        except ValueError:
            pass
        ui.error(f"Please enter numbers between 1 and {len(options)}")


def ask_resource_id(engine: RecommenderEngine, prompt: str = "Resource ID") -> int:
    while True:
        raw = Prompt.ask(f"{prompt} (blank to cancel)", default="")
        if not raw.strip():
            return None
        if raw.strip().isdigit() and engine.get_resource(int(raw)):
            return int(raw)
        ui.error("Unknown resource ID")


# --------------------------------------------------------------------- the app
class CLIApp:
    def __init__(self):
        self.store = ProfileStore()
        self.resources = load_resources()
        self.engine = RecommenderEngine(self.resources, load_ratings())
        self.topics = sorted(self.resources["topic"].unique())
        self.profile = None

    # ------------------------------------------------------------ start screen
    def run(self) -> None:
        ui.banner()
        ui.info(f"Loaded {len(self.resources)} resources and {len(self.engine.ratings)} community ratings.")
        while True:
            ui.menu("Welcome", [("1", "Create new learner profile"), ("2", "Log in"),
                                ("3", "List existing profiles"), ("0", "Exit")])
            choice = Prompt.ask("Choose", choices=["1", "2", "3", "0"], show_choices=False)
            if choice == "1":
                self.create_profile()
            elif choice == "2":
                self.login()
            elif choice == "3":
                users = self.store.list_users()
                ui.info("Profiles: " + (", ".join(users) if users else "none yet"))
            else:
                console.print("[bold cyan]Happy learning! Goodbye.[/bold cyan]")
                return
            if self.profile:
                self.main_menu()

    def create_profile(self) -> None:
        while True:
            username = Prompt.ask("Choose a username (3-30 letters/digits/_/-)").strip()
            if not self.store.valid_username(username):
                ui.error("Invalid username")
            elif self.store.exists(username):
                ui.error("That username already exists - log in instead")
                return
            else:
                break
        profile = UserProfile(username=username.lower(), name=Prompt.ask("Your name", default=""))
        self._edit_preferences(profile)
        self.store.save(profile)
        self.profile = profile
        ui.success(f"Profile '{profile.username}' created. Welcome{', ' + profile.name if profile.name else ''}!")

    def login(self) -> None:
        users = self.store.list_users()
        if not users:
            ui.error("No profiles yet. Create one first.")
            return
        username = Prompt.ask("Username", choices=users, show_choices=True)
        self.profile = self.store.load(username)
        ui.success(f"Logged in as {self.profile.username}")

    def _edit_preferences(self, p: UserProfile) -> None:
        console.print("\n[bold]Your current skill level[/bold]")
        p.level = choose_one("Level", LEVELS, default=LEVELS.index(p.level) + 1)

        console.print("\n[bold]Topics you are interested in[/bold]")
        topics = choose_many("Topics", self.topics, [i for i in p.interests if i in self.topics])
        extra_default = ", ".join(i for i in p.interests if i not in self.topics)
        extra = Prompt.ask("Extra keywords (e.g. pytorch, nlp, sql) comma separated", default=extra_default)
        p.interests = topics + [k.strip() for k in extra.split(",") if k.strip()]

        p.goals = Prompt.ask("Describe your learning goal in a sentence", default=p.goals)

        console.print("\n[bold]Preferred formats[/bold] [dim](blank = any)[/dim]")
        p.preferred_formats = choose_many("Formats", FORMATS, p.preferred_formats)

        limit = FloatPrompt.ask("Maximum hours per resource (0 = no limit)", default=float(p.max_duration or 0))
        p.max_duration = limit if limit > 0 else None
        p.free_only = Confirm.ask("Only show free resources?", default=p.free_only)

    # --------------------------------------------------------------- main menu
    def main_menu(self) -> None:
        actions = {
            "1": ("Get personalized recommendations", self.recommend),
            "2": ("Search resources by keyword", self.search),
            "3": ("Generate a learning path for a goal", self.learning_path),
            "4": ("Find resources similar to one I know", self.similar),
            "5": ("View / rate / complete a resource", self.view_resource),
            "6": ("Browse catalogue by topic", self.browse),
            "7": ("View my profile & history", lambda: ui.profile_panel(self.profile, self.engine)),
            "8": ("Edit my preferences", self.edit_profile),
            "9": ("Evaluate recommender (Precision/Recall@K)", self.evaluate),
            "0": ("Log out", None),
        }
        while self.profile:
            ui.menu(f"Main menu - {self.profile.username}", [(k, v[0]) for k, v in actions.items()])
            choice = Prompt.ask("Choose", choices=list(actions), show_choices=False)
            if choice == "0":
                ui.info(f"Logged out {self.profile.username}")
                self.profile = None
                return
            actions[choice][1]()

    # ----------------------------------------------------------------- actions
    def recommend(self) -> None:
        n = IntPrompt.ask("How many recommendations?", default=5)
        strategy = "hybrid"
        if Confirm.ask("Use advanced options (strategy / topic filter)?", default=False):
            strategy = choose_one("Strategy", STRATEGIES)
            topic = choose_one("Restrict to topic", ["Any"] + self.topics)
            topic = None if topic == "Any" else topic
        else:
            topic = None
        if not self.profile.interests and not self.profile.ratings:
            ui.info("Tip: add interests or rate resources to get more personal results (showing popular picks).")
        recs = self.engine.recommend(self.profile, top_n=max(1, n), strategy=strategy, topic=topic)
        if not recs:
            ui.error("No resources match your constraints. Try relaxing max duration / free-only.")
            return
        ui.recommendations_table(recs, title=f"Top {len(recs)} recommendations ({strategy})")
        self._follow_up({r.resource["id"]: r.breakdown for r in recs})

    def search(self) -> None:
        query = Prompt.ask("Search keywords")
        results = self.engine.search(query, top_n=10)
        if not results:
            ui.error("No results")
            return
        ui.resources_table(results, f"Search results for '{query}'", score_label="Relevance")
        self._follow_up()

    def learning_path(self) -> None:
        goal = Prompt.ask("What do you want to learn? (e.g. 'deep learning', 'web development with python')",
                          default=self.profile.goals or None)
        per_level = IntPrompt.ask("Resources per level", default=2)
        stages = self.engine.learning_path(self.profile, goal, per_level=max(1, per_level))
        ui.learning_path(stages, goal)
        if stages:
            self._follow_up()

    def similar(self) -> None:
        rid = ask_resource_id(self.engine, "ID of a resource you liked")
        if rid is None:
            return
        base = self.engine.get_resource(rid)
        ui.resources_table(self.engine.similar_resources(rid, 5), f"Resources similar to '{base['title']}'",
                           score_label="Similarity")
        self._follow_up()

    def browse(self) -> None:
        topic = choose_one("Topic", ["All"] + self.topics)
        df = self.resources if topic == "All" else self.resources[self.resources["topic"] == topic]
        df = df.sort_values(["level", "rating"], key=lambda s: s.map({l: i for i, l in enumerate(LEVELS)})
                            if s.name == "level" else -s)
        ui.resources_table(df.to_dict("records"), f"Catalogue - {topic}")
        self._follow_up()

    def _follow_up(self, breakdowns: dict = None) -> None:
        while True:
            rid = ask_resource_id(self.engine, "Enter an ID to open details/rate")
            if rid is None:
                return
            self._resource_actions(rid, (breakdowns or {}).get(rid))

    def view_resource(self) -> None:
        rid = ask_resource_id(self.engine)
        if rid is not None:
            self._resource_actions(rid)

    def _resource_actions(self, rid: int, breakdown: dict = None) -> None:
        p = self.profile
        ui.resource_detail(self.engine.get_resource(rid), self.engine.community_stats(rid),
                           p.ratings.get(rid), rid in p.completed, breakdown)
        choice = Prompt.ask("\\[r]ate  \\[c]omplete  \\[s]imilar  \\[b]ack", choices=["r", "c", "s", "b"], default="b")
        if choice == "r":
            rating = IntPrompt.ask("Your rating 1-5", choices=["1", "2", "3", "4", "5"])
            p.rate(rid, rating)
            self.engine.add_rating(p.username, rid, rating)   # model learns immediately
            save_ratings(self.engine.ratings)
            self.store.save(p)
            ui.success("Thanks! Your future recommendations will adapt to this rating.")
        elif choice == "c":
            p.mark_completed(rid)
            self.store.save(p)
            ui.success("Marked as completed.")
            if rid not in p.ratings and Confirm.ask("Would you like to rate it?", default=True):
                rating = IntPrompt.ask("Your rating 1-5", choices=["1", "2", "3", "4", "5"])
                p.rate(rid, rating)
                self.engine.add_rating(p.username, rid, rating)
                save_ratings(self.engine.ratings)
                self.store.save(p)
                ui.success("Rating saved.")
        elif choice == "s":
            ui.resources_table(self.engine.similar_resources(rid, 5), "Similar resources", score_label="Similarity")

    def edit_profile(self) -> None:
        self.profile.name = Prompt.ask("Your name", default=self.profile.name)
        self._edit_preferences(self.profile)
        self.store.save(self.profile)
        ui.success("Preferences updated.")

    def evaluate(self) -> None:
        k = IntPrompt.ask("K (number of recommendations to evaluate)", default=5)
        with console.status("Running hold-out evaluation..."):
            df = evaluate(self.resources, self.engine.ratings, k=max(1, k))
        from rich.table import Table
        table = Table(title=f"Offline evaluation ({df.attrs['users_evaluated']} learners, 30% liked items hidden)",
                      header_style="bold magenta")
        for col in df.columns:
            table.add_column(col, justify="right" if col != "strategy" else "left")
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)
        ui.info("Higher is better. 'random' is the baseline every model should beat.")
