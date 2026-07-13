import pytest

from llm_plan.dag import (
    leaf_stages,
    resolve_dependencies,
    topological_order,
    validate_plan,
)
from llm_plan.models import Plan, PlanError, Stage


def make_stage(name, prompt="CLI", **kwargs):
    return Stage(name=name, summary=f"{name} stage", model="echo", prompt=prompt, **kwargs)


def make_plan(*stages):
    return Plan(path=None, stages=list(stages))


class TestResolveDependencies:
    def test_explicit_depends_on_wins(self):
        stages = [make_stage("a"), make_stage("b", depends_on=["a"])]

        assert resolve_dependencies(stages[1], 1, stages) == ["a"]

    def test_non_first_stage_without_cli_prompt_implicitly_depends_on_previous(self):
        stages = [make_stage("a"), make_stage("b", prompt="inline:continue")]

        assert resolve_dependencies(stages[1], 1, stages) == ["a"]

    def test_cli_prompt_breaks_the_implicit_chain(self):
        stages = [make_stage("a"), make_stage("b", prompt="CLI")]

        assert resolve_dependencies(stages[1], 1, stages) == []

    def test_first_stage_has_no_implicit_dependency(self):
        stages = [make_stage("a", prompt="inline:go")]

        assert resolve_dependencies(stages[0], 0, stages) == []

    def test_chain_targets_join_the_scheduling_dependencies(self):
        stages = [
            make_stage("a"),
            make_stage("b", prompt="CLI"),
            make_stage("c", depends_on=["b"], prompt=[{"prompt": "chain:a", "label": "A"}]),
        ]

        assert resolve_dependencies(stages[2], 2, stages) == ["b", "a"]

    def test_a_chain_target_already_depended_on_is_not_duplicated(self):
        stages = [
            make_stage("a"),
            make_stage("b", depends_on=["a"], prompt="chain:a"),
        ]

        assert resolve_dependencies(stages[1], 1, stages) == ["a"]

    def test_a_forward_chain_reference_is_rejected_like_a_forward_dependency(self):
        stages = [
            make_stage("a", prompt=[{"prompt": "chain:b", "label": "B"}]),
            make_stage("b", prompt="CLI"),
        ]

        with pytest.raises(PlanError, match="appears later"):
            validate_plan(make_plan(*stages))

    def test_a_chained_stage_is_not_a_leaf(self):
        stages = [
            make_stage("a"),
            make_stage("b", prompt="CLI"),
            make_stage("c", depends_on=["b"], prompt=[{"prompt": "chain:a", "label": "A"}]),
        ]

        assert leaf_stages(stages) == ["c"]


class TestValidatePlan:
    def test_accepts_a_valid_diamond(self):
        validate_plan(
            make_plan(
                make_stage("seed_reader", depends_on=["seed"]),
                make_stage("left", depends_on=["seed_reader"]),
                make_stage("right", depends_on=["seed_reader"]),
                make_stage("join", depends_on=["left", "right"]),
            )
        )

    def test_unknown_dependency_is_an_error(self):
        with pytest.raises(PlanError, match="unknown stage 'ghost'"):
            validate_plan(make_plan(make_stage("a", depends_on=["ghost"])))

    def test_dependency_listed_after_dependent_is_an_error(self):
        with pytest.raises(PlanError, match="appears later"):
            validate_plan(
                make_plan(
                    make_stage("a", depends_on=["b"]),
                    make_stage("b"),
                )
            )

    def test_stage_named_seed_is_rejected_as_reserved(self):
        with pytest.raises(PlanError, match="'seed' is reserved"):
            validate_plan(make_plan(make_stage("seed")))

    def test_duplicate_stage_names_are_an_error(self):
        with pytest.raises(PlanError, match="[Dd]uplicate.*twin"):
            validate_plan(make_plan(make_stage("twin"), make_stage("twin")))

    @pytest.mark.parametrize(
        "prompt",
        ["chain:ghost", [{"prompt": "chain:ghost", "label": "Analysis"}]],
        ids=["scalar", "list-mapping"],
    )
    def test_chain_to_an_unknown_stage_is_an_error(self, prompt):
        with pytest.raises(PlanError, match="'follow_up'.*chain:'ghost'"):
            validate_plan(
                make_plan(
                    make_stage("analyst"),
                    make_stage("follow_up", prompt=prompt, depends_on=["analyst"]),
                )
            )

    def test_chain_to_a_known_stage_is_accepted(self):
        validate_plan(
            make_plan(
                make_stage("analyst"),
                make_stage(
                    "follow_up", prompt="chain:analyst", depends_on=["analyst"]
                ),
            )
        )

    def test_self_dependency_is_an_error(self):
        with pytest.raises(PlanError, match="'a' depends on 'a'"):
            validate_plan(make_plan(make_stage("a", depends_on=["a"])))


class TestTopologicalOrder:
    def test_orders_dependencies_first_with_listed_order_tiebreak(self):
        stages = [
            make_stage("analyst"),
            make_stage("reviewer"),
            make_stage("synthesis", depends_on=["analyst", "reviewer"]),
        ]

        assert topological_order(stages) == ["analyst", "reviewer", "synthesis"]

    def test_seed_is_not_part_of_the_order(self):
        stages = [make_stage("a", depends_on=["seed"]), make_stage("b", depends_on=["a"])]

        assert topological_order(stages) == ["a", "b"]


class TestLeafStages:
    def test_stages_with_no_dependents_are_leaves(self):
        stages = [
            make_stage("analyst"),
            make_stage("reviewer"),
            make_stage("synthesis", depends_on=["analyst", "reviewer"]),
        ]

        assert leaf_stages(stages) == ["synthesis"]

    def test_implicit_dependencies_count(self):
        stages = [make_stage("a"), make_stage("b", prompt="inline:continue")]

        assert leaf_stages(stages) == ["b"]

    def test_independent_stages_are_all_leaves(self):
        stages = [make_stage("a"), make_stage("b")]

        assert leaf_stages(stages) == ["a", "b"]
