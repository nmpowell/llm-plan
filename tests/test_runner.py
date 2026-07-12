import llm
import pytest
import yaml

from llm_plan.models import PlanError
from llm_plan.runner import CLIContext, PlanRunner, load_plan


def write_plan(tmp_path, stages, **top_level):
    data = {"name": "test", "summary": "a test plan", "stages": stages, **top_level}
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return plan_file


def run_plan(tmp_path, stages, cli=None, **runner_kwargs):
    """Run a plan; sequential-abort errors are tolerated so results stay inspectable."""
    plan = load_plan(write_plan(tmp_path, stages))
    runner = PlanRunner(plan, cli or CLIContext(), retry_delay=0, **runner_kwargs)
    try:
        runner.run()
    except PlanError:
        pass
    return runner, dict(runner.results)


class TestSingleStage:
    def test_cli_stage_sends_instructions_and_returns_text(self, tmp_path, fake_models):
        runner, results = run_plan(
            tmp_path,
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}],
            CLIContext(instructions="What is love?"),
        )

        assert results["solo"].success
        assert results["solo"].text == "ECHO[What is love?]"
        assert results["solo"].response_id

    def test_stages_stream_their_responses(self, tmp_path, fake_models):
        # Anthropic's SDK rejects non-streaming requests whose max_tokens
        # implies a long run; llm streams by default and so must plan stages.
        run_plan(
            tmp_path,
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}],
            CLIContext(instructions="hi"),
        )

        assert fake_models.echo.stream_flags == [True]

    def test_non_streaming_models_are_called_without_streaming(self, tmp_path):
        stages = [{"name": "solo", "summary": "s", "model": "nostream", "prompt": "CLI"}]

        runner, results = run_plan(
            tmp_path, stages, CLIContext(instructions="hi")
        )

        assert results["solo"].success, results["solo"].error
        assert results["solo"].text == "NOSTREAM-OK"

    def test_negative_retries_are_rejected_at_construction(self, tmp_path):
        plan = load_plan(write_plan(
            tmp_path, [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]
        ))

        with pytest.raises(PlanError, match="retries"):
            PlanRunner(plan, CLIContext(), retries=-1)

    def test_plain_cli_stage_with_no_cli_content_fails_helpfully(self, tmp_path):
        runner, results = run_plan(
            tmp_path,
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}],
            CLIContext(),
        )

        assert not results["solo"].success
        assert "prompt" in results["solo"].error and "-f" in results["solo"].error

    def test_unknown_model_fails_the_stage_with_llms_message(self, tmp_path):
        runner, results = run_plan(
            tmp_path,
            [{"name": "solo", "summary": "s", "model": "no-such-model", "prompt": "CLI"}],
            CLIContext(instructions="hi"),
        )

        assert not results["solo"].success
        assert "no-such-model" in results["solo"].error

    def test_invalid_option_fails_the_stage_without_retrying(self, tmp_path, fake_models):
        stage = {
            "name": "solo",
            "summary": "s",
            "model": "echo",
            "prompt": "CLI",
            "options": {"nonsense_option": 1},
        }

        runner, results = run_plan(tmp_path, [stage], CLIContext(instructions="hi"))

        assert not results["solo"].success
        assert "nonsense_option" in results["solo"].error
        assert len(fake_models.echo.prompts) == 0

    def test_empty_composed_prompt_is_a_stage_failure(self, tmp_path):
        runner, results = run_plan(
            tmp_path,
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI:instructions"}],
            CLIContext(),
        )

        assert not results["solo"].success
        assert "empty" in results["solo"].error.lower()


class TestPromptComposition:
    def test_dependency_output_is_a_labelled_section(self, tmp_path):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI",
             "produces": "First Answer"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="go"))

        text = results["second"].text
        assert "## First Answer" in text
        assert "ECHO[go]" in text
        assert "Continue." in text

    def test_dependency_without_produces_gets_a_default_label(self, tmp_path):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="go"))

        assert "## Output from first" in results["second"].text

    def test_chain_prompt_injects_upstream_text_with_label(self, tmp_path):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo", "depends_on": ["first"],
             "prompt": [{"prompt": "chain:first", "label": "Earlier Analysis"}]},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="go"))

        assert "## Earlier Analysis" in results["second"].text

    def test_stage_files_and_prompt_files_are_labelled_sections(self, tmp_path):
        (tmp_path / "ctx.md").write_text("the context", encoding="utf-8")
        (tmp_path / "guide.md").write_text("the guide", encoding="utf-8")
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": ["guide.md"],
             "files": [{"path": "ctx.md", "label": "Context"}]},
        ]

        runner, results = run_plan(tmp_path, stages)

        text = results["solo"].text
        assert "## Context" in text and "the context" in text
        assert "## MAIN INSTRUCTIONS" in text and "the guide" in text
        assert text.index("the context") < text.index("the guide")

    def test_prompt_label_renames_the_first_prompt_file(self, tmp_path):
        (tmp_path / "guide.md").write_text("the guide", encoding="utf-8")
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": "guide.md", "prompt_label": "Instructions"},
        ]

        runner, results = run_plan(tmp_path, stages)

        assert "## Instructions" in results["solo"].text

    def test_labelled_cli_instructions_get_their_heading(self, tmp_path):
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": [{"prompt": "CLI:instructions", "label": "Original Question"}]},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="why?"))

        assert "## Original Question" in results["solo"].text
        assert "why?" in results["solo"].text

    def test_headed_instruction_parts_render_with_their_headings(self, tmp_path):
        stages = [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]
        cli = CLIContext(
            instructions="the main question",
            instruction_parts=[("Focus Areas", "look at concurrency")],
        )

        runner, results = run_plan(tmp_path, stages, cli)

        text = results["solo"].text
        assert "## Focus Areas" in text
        assert "look at concurrency" in text
        assert "the main question" in text
        assert text.index("look at concurrency") < text.index("the main question")


class TestGoldenComposition:
    def test_composed_prompt_matches_the_legacy_shape_exactly(self, tmp_path, fake_models):
        (tmp_path / "ctx.md").write_text("context body\n", encoding="utf-8")
        (tmp_path / "guide.md").write_text("guide body\n", encoding="utf-8")
        stages = [
            {"name": "analyst", "summary": "s", "model": "echo", "prompt": "CLI",
             "produces": "Analysis"},
            {"name": "synthesis", "summary": "s", "model": "echo",
             "depends_on": ["analyst"],
             "files": [{"path": "ctx.md", "label": "Context"}],
             "prompt": [
                 {"prompt": "CLI:instructions", "label": "Original Question"},
                 {"prompt": "guide.md", "label": "Guidelines"},
                 "inline:Also consider tone.",
             ]},
        ]

        run_plan(tmp_path, stages, CLIContext(instructions="the question"))

        expected = (
            "---\n\n## Context\n\n```\ncontext body\n```\n"
            "\n"
            "---\n\n## Analysis\n\n```\nECHO[the question]\n```\n"
            "\n"
            "---\n\n## Guidelines\n\n```\nguide body\n```\n"
            "\n"
            "---\n\n## Original Question\n\nthe question\n"
            "\n"
            "---\n\n## Additional Instructions\n\nAlso consider tone.\n"
        )
        assert fake_models.echo.prompts[-1].prompt == expected

    def test_a_lone_cli_prompt_stays_raw(self, tmp_path, fake_models):
        stages = [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]

        run_plan(tmp_path, stages, CLIContext(instructions="just the question"))

        assert fake_models.echo.prompts[0].prompt == "just the question"


class TestCliContextRouting:
    def test_fragments_reach_cli_stages_but_not_downstream_stages(self, tmp_path, fake_models):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]
        cli = CLIContext(instructions="go", fragments=["seed fragment text"])

        runner, results = run_plan(tmp_path, stages, cli)

        assert "seed fragment text" in results["first"].text
        first_prompt, second_prompt = fake_models.echo.prompts
        assert first_prompt.fragments == ["seed fragment text"]
        assert second_prompt.fragments == []

    def test_cli_instructions_variant_excludes_fragments(self, tmp_path, fake_models):
        stages = [
            {"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI:instructions"},
        ]
        cli = CLIContext(instructions="go", fragments=["seed fragment text"])

        runner, results = run_plan(tmp_path, stages, cli)

        assert fake_models.echo.prompts[0].fragments == []

    def test_seed_dependency_grants_cli_context(self, tmp_path, fake_models):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "late", "summary": "s", "model": "echo",
             "depends_on": ["seed"], "prompt": "inline:Look at the fragments."},
        ]
        cli = CLIContext(instructions="go", fragments=["seed fragment text"])

        runner, results = run_plan(tmp_path, stages, cli)

        late_prompt = fake_models.echo.prompts[-1]
        assert late_prompt.fragments == ["seed fragment text"]

    def test_first_stage_without_prompt_gets_cli_files_automatically(self, tmp_path, fake_models):
        stages = [
            {"name": "solo", "summary": "s", "model": "echo", "prompt": "inline:Describe input."},
        ]
        cli = CLIContext(fragments=["seed fragment text"])

        runner, results = run_plan(tmp_path, stages, cli)

        assert fake_models.echo.prompts[0].fragments == ["seed fragment text"]

    def test_attachments_follow_the_same_routing(self, tmp_path, fake_models, png_bytes):
        image = tmp_path / "img.png"
        image.write_bytes(png_bytes)
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]
        cli = CLIContext(instructions="go", attachments=[llm.Attachment(path=str(image))])

        runner, results = run_plan(tmp_path, stages, cli)

        first_prompt, second_prompt = fake_models.echo.prompts
        assert [a.path for a in first_prompt.attachments] == [str(image)]
        assert second_prompt.attachments == []


class TestModelsAndOptions:
    def test_each_stage_uses_its_own_model(self, tmp_path, fake_models):
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "b", "summary": "s", "model": "other", "prompt": "CLI"},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        assert results["a"].text.startswith("ECHO[")
        assert results["b"].text.startswith("OTHER[")

    def test_cli_model_overrides_every_stage(self, tmp_path, fake_models):
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "b", "summary": "s", "model": "other", "prompt": "CLI"},
        ]
        cli = CLIContext(instructions="hi", model="other")

        runner, results = run_plan(tmp_path, stages, cli)

        assert results["a"].text.startswith("OTHER[")
        assert results["b"].text.startswith("OTHER[")

    def test_cli_options_override_stage_options(self, tmp_path, fake_models):
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI",
             "options": {"temperature": 0.0, "max_tokens": 100}},
        ]
        cli = CLIContext(instructions="hi", options={"temperature": 0.7})

        runner, results = run_plan(tmp_path, stages, cli)

        options = fake_models.echo.prompts[0].options
        assert options.temperature == 0.7
        assert options.max_tokens == 100


class TestFailureHandling:
    def test_transient_failure_is_retried_then_succeeds(self, tmp_path, fake_models):
        fake_models.flaky.failures_left = 2
        stages = [{"name": "solo", "summary": "s", "model": "flaky", "prompt": "CLI"}]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        assert results["solo"].success
        assert fake_models.flaky.calls == 3

    def test_failure_after_retries_exhausted_fails_the_stage(self, tmp_path, fake_models):
        fake_models.flaky.failures_left = 10
        stages = [{"name": "solo", "summary": "s", "model": "flaky", "prompt": "CLI"}]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        assert not results["solo"].success
        assert "transient upstream error" in results["solo"].error
        assert fake_models.flaky.calls == 3

    def test_intolerant_dependents_after_a_tolerated_failure_are_skipped(
        self, tmp_path, fake_models
    ):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "rescue", "summary": "s", "model": "echo", "prompt": "CLI",
             "depends_on": ["bad"], "partial_dependencies": True},
            {"name": "after", "summary": "s", "model": "echo",
             "depends_on": ["bad"], "prompt": "inline:Continue."},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        assert results["rescue"].success
        assert not results["after"].success
        assert "bad" in results["after"].error

    def test_partial_dependencies_runs_with_surviving_outputs(self, tmp_path, fake_models):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "good", "summary": "s", "model": "echo", "prompt": "CLI",
             "produces": "Good Analysis"},
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "synthesis", "summary": "s", "model": "echo",
             "depends_on": ["bad", "good"], "partial_dependencies": True,
             "prompt": "inline:Synthesise."},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        assert results["synthesis"].success
        assert "## Good Analysis" in results["synthesis"].text
        assert "bad" not in results["synthesis"].text.lower().replace("flaky-ok", "")

    def test_sequential_failure_aborts_before_later_stages_spend_money(
        self, tmp_path, fake_models
    ):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "independent", "summary": "s", "model": "echo", "prompt": "CLI"},
        ]
        plan = load_plan(write_plan(tmp_path, stages))
        runner = PlanRunner(plan, CLIContext(instructions="hi"), retry_delay=0)

        with pytest.raises(PlanError, match="bad"):
            runner.run()

        assert fake_models.echo.prompts == []

    def test_sequential_failure_continues_when_the_next_stage_tolerates_it(
        self, tmp_path, fake_models
    ):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "rescue", "summary": "s", "model": "echo", "prompt": "CLI",
             "depends_on": ["bad"], "partial_dependencies": True},
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        assert not results["bad"].success
        assert results["rescue"].success


class TestResponses:
    def test_on_response_fires_for_each_successful_llm_stage(self, tmp_path):
        seen = []
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "b", "summary": "s", "model": "echo", "prompt": "CLI"},
        ]
        plan = load_plan(write_plan(tmp_path, stages))
        runner = PlanRunner(
            plan,
            CLIContext(instructions="hi"),
            retry_delay=0,
            on_response=lambda stage, response: seen.append((stage.name, response)),
        )

        runner.run()

        assert [name for name, _ in seen] == ["a", "b"]
        assert all(response.text() for _, response in seen)


class TestLoadPlan:
    def test_load_plan_validates_the_dag(self, tmp_path):
        plan_file = write_plan(
            tmp_path,
            [{"name": "a", "summary": "s", "model": "echo", "prompt": "CLI",
              "depends_on": ["ghost"]}],
        )

        with pytest.raises(PlanError, match="ghost"):
            load_plan(plan_file)

    def test_load_plan_rejects_unknown_runtime_variables(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("", encoding="utf-8")
        plan_file = write_plan(
            tmp_path,
            [{"name": "a", "summary": "s", "type": "python_script", "script": "s.py",
              "script_args": ["${cli.nonsense}"]}],
        )

        with pytest.raises(PlanError, match="cli.nonsense"):
            load_plan(plan_file)
