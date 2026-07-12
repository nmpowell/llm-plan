import pytest
import yaml

from llm_plan.models import PlanError
from llm_plan.store import is_alias, list_plans, resolve_plan, user_plan_dir


def write_plan(directory, filename, name="p", summary="a plan"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(yaml.safe_dump({"name": name, "summary": summary}), encoding="utf-8")
    return path


class TestIsAlias:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("synthesis_full", True),
            ("plan.yaml", False),
            ("plans/synthesis.yaml", False),
            ("./plan", False),
            ("~/plans/x", False),
            ("sub\\plan", False),
        ],
    )
    def test_paths_are_not_aliases(self, value, expected):
        assert is_alias(value) is expected


class TestResolvePlan:
    def test_explicit_path_resolves_directly(self, tmp_path):
        plan = write_plan(tmp_path, "my_plan.yaml")

        assert resolve_plan(str(plan)) == plan

    def test_missing_explicit_path_is_an_error(self, tmp_path):
        with pytest.raises(PlanError, match="not found"):
            resolve_plan(str(tmp_path / "gone.yaml"))

    def test_alias_finds_plan_prefixed_file_in_user_dir(self, user_dir):
        plan = write_plan(user_dir / "plans", "plan_synthesis.yaml")

        assert resolve_plan("synthesis") == plan

    def test_alias_falls_back_to_unprefixed_name(self, user_dir):
        plan = write_plan(user_dir / "plans", "review.yaml")

        assert resolve_plan("review") == plan

    def test_env_dirs_take_precedence_over_user_dir(self, user_dir, tmp_path, monkeypatch):
        env_dir = tmp_path / "team-plans"
        env_plan = write_plan(env_dir, "plan_shared.yaml", summary="env copy")
        write_plan(user_dir / "plans", "plan_shared.yaml", summary="user copy")
        monkeypatch.setenv("LLM_PLAN_DIRS", str(env_dir))

        assert resolve_plan("shared") == env_plan

    def test_unknown_alias_error_names_the_searched_locations(self, user_dir):
        with pytest.raises(PlanError, match="plan_nope.yaml"):
            resolve_plan("nope")

    def test_bundled_plans_resolve(self):
        path = resolve_plan("synthesis_full")

        assert path.name == "plan_synthesis_full.yaml"
        assert path.exists()

    def test_user_plan_shadows_a_bundled_alias(self, user_dir):
        mine = write_plan(user_dir / "plans", "plan_synthesis_full.yaml")

        assert resolve_plan("synthesis_full") == mine


class TestListPlans:
    def test_lists_alias_name_summary_and_path(self, user_dir):
        write_plan(user_dir / "plans", "plan_review.yaml", name="review", summary="reviews things")

        listings = {p.alias: p for p in list_plans()}

        assert "review" in listings
        assert listings["review"].summary == "reviews things"
        assert listings["review"].path == user_plan_dir() / "plan_review.yaml"

    def test_underscore_prefixed_files_are_hidden(self, user_dir):
        write_plan(user_dir / "plans", "_defaults.yaml")

        assert "defaults" not in {p.alias for p in list_plans()}
        assert "_defaults" not in {p.alias for p in list_plans()}

    def test_first_match_wins_across_locations(self, user_dir, tmp_path, monkeypatch):
        env_dir = tmp_path / "team-plans"
        write_plan(env_dir, "plan_x.yaml", summary="env copy")
        write_plan(user_dir / "plans", "plan_x.yaml", summary="user copy")
        monkeypatch.setenv("LLM_PLAN_DIRS", str(env_dir))

        listings = {p.alias: p for p in list_plans()}

        assert listings["x"].summary == "env copy"

    def test_bundled_synthesis_full_is_listed(self):
        assert "synthesis_full" in {p.alias for p in list_plans()}

    def test_unreadable_yaml_still_lists_with_empty_summary(self, user_dir):
        plans = user_plan_dir()
        (plans / "plan_broken.yaml").write_text(": not: valid: yaml [", encoding="utf-8")

        listings = {p.alias: p for p in list_plans()}

        assert listings["broken"].summary == ""
