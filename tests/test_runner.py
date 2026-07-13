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


class TestSingleStage:
    def test_cli_stage_sends_instructions_and_returns_text(self, tmp_path, fake_models, run_plan):
        runner, results = run_plan(
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}],
            CLIContext(instructions="What is love?"),
        )

        assert results["solo"].success
        assert results["solo"].text == "ECHO[What is love?]"
        assert results["solo"].response_id

    def test_stages_stream_their_responses(self, tmp_path, fake_models, run_plan):
        # Anthropic's SDK rejects non-streaming requests whose max_tokens
        # implies a long run; llm streams by default and so must plan stages.
        run_plan(
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}],
            CLIContext(instructions="hi"),
        )

        assert fake_models.echo.stream_flags == [True]

    def test_non_streaming_models_are_called_without_streaming(self, tmp_path, run_plan):
        stages = [{"name": "solo", "summary": "s", "model": "nostream", "prompt": "CLI"}]

        runner, results = run_plan(
            stages, CLIContext(instructions="hi")
        )

        assert results["solo"].success, results["solo"].error
        assert results["solo"].text == "NOSTREAM-OK"

    def test_negative_retries_are_rejected_at_construction(self, tmp_path):
        plan = load_plan(write_plan(
            tmp_path, [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]
        ))

        with pytest.raises(PlanError, match="retries"):
            PlanRunner(plan, CLIContext(), retries=-1)

    def test_plain_cli_stage_with_no_cli_content_fails_helpfully(self, tmp_path, run_plan):
        runner, results = run_plan(
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}],
            CLIContext(),
            expect_error=True,
        )

        assert not results["solo"].success
        assert "prompt" in results["solo"].error and "-f" in results["solo"].error

    def test_unknown_model_fails_the_stage_with_llms_message(self, tmp_path, run_plan):
        runner, results = run_plan(
            [{"name": "solo", "summary": "s", "model": "no-such-model", "prompt": "CLI"}],
            CLIContext(instructions="hi"),
            expect_error=True,
        )

        assert not results["solo"].success
        assert "no-such-model" in results["solo"].error

    def test_invalid_option_fails_the_stage_without_retrying(self, tmp_path, fake_models, run_plan):
        stage = {
            "name": "solo",
            "summary": "s",
            "model": "echo",
            "prompt": "CLI",
            "options": {"nonsense_option": 1},
        }

        runner, results = run_plan([stage], CLIContext(instructions="hi"), expect_error=True)

        assert not results["solo"].success
        assert "nonsense_option" in results["solo"].error
        assert len(fake_models.echo.prompts) == 0

    def test_empty_composed_prompt_is_a_stage_failure(self, tmp_path, run_plan):
        runner, results = run_plan(
            [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI:instructions"}],
            CLIContext(),
            expect_error=True,
        )

        assert not results["solo"].success
        assert "empty" in results["solo"].error.lower()


class TestPromptComposition:
    def test_dependency_output_is_a_labelled_section(self, tmp_path, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI",
             "produces": "First Answer"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="go"))

        text = results["second"].text
        assert "## First Answer" in text
        assert "ECHO[go]" in text
        assert "Continue." in text

    def test_dependency_without_produces_gets_a_default_label(self, tmp_path, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="go"))

        assert "## Output from first" in results["second"].text

    def test_chain_prompt_injects_upstream_text_with_label(self, tmp_path, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo", "depends_on": ["first"],
             "prompt": [{"prompt": "chain:first", "label": "Earlier Analysis"}]},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="go"))

        assert "## Earlier Analysis" in results["second"].text

    def test_a_chain_consumed_dependency_is_not_also_auto_fenced(self, tmp_path, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo", "depends_on": ["first"],
             "prompt": [{"prompt": "chain:first", "label": "Earlier Analysis"}]},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="go"))

        # The chain: item places first's text; a second auto-fenced copy
        # would send (and bill) the same content twice.
        text = results["second"].text
        assert "## Output from first" not in text
        assert text.count("ECHO[go]") == 1

    def test_a_chain_only_reference_still_receives_the_dependency_text(
        self, tmp_path, run_plan
    ):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "detour", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "third", "summary": "s", "model": "echo", "depends_on": ["detour"],
             "prompt": [{"prompt": "chain:first", "label": "First Take"}]},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="go"))

        text = results["third"].text
        assert "## First Take" in text
        assert "## Output from detour" in text

    def test_stage_files_and_prompt_files_are_labelled_sections(self, tmp_path, run_plan):
        (tmp_path / "ctx.md").write_text("the context", encoding="utf-8")
        (tmp_path / "guide.md").write_text("the guide", encoding="utf-8")
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": ["guide.md"],
             "files": [{"path": "ctx.md", "label": "Context"}]},
        ]

        runner, results = run_plan(stages)

        text = results["solo"].text
        assert "## Context" in text and "the context" in text
        assert "## MAIN INSTRUCTIONS" in text and "the guide" in text
        assert text.index("the context") < text.index("the guide")

    def test_a_binary_context_file_fails_naming_the_stage_and_file(self, tmp_path, run_plan):
        (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": "inline:Describe the input.",
             "files": [{"path": "binary.bin"}]},
        ]

        runner, results = run_plan(stages, expect_error=True)

        assert not results["solo"].success
        assert "binary.bin" in results["solo"].error
        assert "solo" in results["solo"].error

    def test_prompt_label_renames_the_first_prompt_file(self, tmp_path, run_plan):
        (tmp_path / "guide.md").write_text("the guide", encoding="utf-8")
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": "guide.md", "prompt_label": "Instructions"},
        ]

        runner, results = run_plan(stages)

        assert "## Instructions" in results["solo"].text

    def test_labelled_cli_instructions_get_their_heading(self, tmp_path, run_plan):
        stages = [
            {"name": "solo", "summary": "s", "model": "echo",
             "prompt": [{"prompt": "CLI:instructions", "label": "Original Question"}]},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="why?"))

        assert "## Original Question" in results["solo"].text
        assert "why?" in results["solo"].text

    def test_headed_instruction_parts_render_with_their_headings(self, tmp_path, run_plan):
        stages = [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]
        cli = CLIContext(
            instructions="the main question",
            instruction_parts=[("Focus Areas", "look at concurrency")],
        )

        runner, results = run_plan(stages, cli)

        text = results["solo"].text
        assert "## Focus Areas" in text
        assert "look at concurrency" in text
        assert "the main question" in text
        assert text.index("look at concurrency") < text.index("the main question")


class TestGoldenComposition:
    def test_composed_prompt_matches_the_pinned_shape_exactly(self, tmp_path, fake_models, run_plan):
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

        run_plan(stages, CLIContext(instructions="the question"))

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

    def test_a_lone_cli_prompt_stays_raw(self, tmp_path, fake_models, run_plan):
        stages = [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]

        run_plan(stages, CLIContext(instructions="just the question"))

        assert fake_models.echo.prompts[0].prompt == "just the question"

    def test_a_lone_cli_prompt_keeps_its_whitespace(self, tmp_path, fake_models, run_plan):
        piped_code = "    def f():\n        return 1\n"

        stages = [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]
        run_plan(stages, CLIContext(instructions=piped_code))

        assert fake_models.echo.prompts[0].prompt == piped_code


class TestCliContextRouting:
    def test_fragments_reach_cli_stages_but_not_downstream_stages(self, tmp_path, fake_models, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]
        cli = CLIContext(instructions="go", fragments=["seed fragment text"])

        runner, results = run_plan(stages, cli)

        assert "seed fragment text" in results["first"].text
        first_prompt, second_prompt = fake_models.echo.prompts
        assert first_prompt.fragments == ["seed fragment text"]
        assert second_prompt.fragments == []

    def test_cli_instructions_variant_excludes_fragments(self, tmp_path, fake_models, run_plan):
        stages = [
            {"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI:instructions"},
        ]
        cli = CLIContext(instructions="go", fragments=["seed fragment text"])

        runner, results = run_plan(stages, cli)

        assert fake_models.echo.prompts[0].fragments == []

    def test_seed_dependency_grants_cli_context(self, tmp_path, fake_models, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "late", "summary": "s", "model": "echo",
             "depends_on": ["seed"], "prompt": "inline:Look at the fragments."},
        ]
        cli = CLIContext(instructions="go", fragments=["seed fragment text"])

        runner, results = run_plan(stages, cli)

        late_prompt = fake_models.echo.prompts[-1]
        assert late_prompt.fragments == ["seed fragment text"]

    def test_first_stage_without_a_cli_prompt_still_receives_seed_files(
        self, tmp_path, fake_models
    , run_plan):
        stages = [
            {"name": "solo", "summary": "s", "model": "echo", "prompt": "inline:Describe input."},
        ]
        cli = CLIContext(fragments=["seed fragment text"])

        runner, results = run_plan(stages, cli)

        assert fake_models.echo.prompts[0].fragments == ["seed fragment text"]

    def test_attachments_follow_the_same_routing(self, tmp_path, fake_models, png_bytes, run_plan):
        image = tmp_path / "img.png"
        image.write_bytes(png_bytes)
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "second", "summary": "s", "model": "echo",
             "depends_on": ["first"], "prompt": "inline:Continue."},
        ]
        cli = CLIContext(instructions="go", attachments=[llm.Attachment(path=str(image))])

        runner, results = run_plan(stages, cli)

        first_prompt, second_prompt = fake_models.echo.prompts
        assert [a.path for a in first_prompt.attachments] == [str(image)]
        assert second_prompt.attachments == []


class TestSeedRoutingWarnings:
    def test_warns_when_no_stage_will_receive_cli_files(self, tmp_path, run_plan):
        messages = []
        stages = [
            {"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI:instructions"},
        ]
        cli = CLIContext(instructions="hi", fragments=["seed fragment text"])

        run_plan(stages, cli, progress=messages.append)

        assert any("no stage requests CLI input" in m for m in messages), messages

    def test_no_warning_when_a_stage_receives_cli_files(self, tmp_path, run_plan):
        messages = []
        stages = [{"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI"}]
        cli = CLIContext(instructions="hi", fragments=["seed fragment text"])

        run_plan(stages, cli, progress=messages.append)

        assert not any("no stage requests CLI input" in m for m in messages)

    def test_no_warning_when_only_instructions_were_provided(self, tmp_path, run_plan):
        messages = []
        stages = [
            {"name": "solo", "summary": "s", "model": "echo", "prompt": "CLI:instructions"},
        ]

        run_plan(stages, CLIContext(instructions="hi"), progress=messages.append)

        assert not any("no stage requests CLI input" in m for m in messages)


class TestModelsAndOptions:
    def test_each_stage_uses_its_own_model(self, tmp_path, fake_models, run_plan):
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "b", "summary": "s", "model": "other", "prompt": "CLI"},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="hi"))

        assert results["a"].text.startswith("ECHO[")
        assert results["b"].text.startswith("OTHER[")

    def test_cli_model_overrides_every_stage(self, tmp_path, fake_models, run_plan):
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI"},
            {"name": "b", "summary": "s", "model": "other", "prompt": "CLI"},
        ]
        cli = CLIContext(instructions="hi", model="other")

        runner, results = run_plan(stages, cli)

        assert results["a"].text.startswith("OTHER[")
        assert results["b"].text.startswith("OTHER[")

    def test_invalid_options_are_reported_without_pydantic_boilerplate(
        self, tmp_path, run_plan
    ):
        stage = {
            "name": "solo",
            "summary": "s",
            "model": "echo",
            "prompt": "CLI",
            "options": {"nonsense_option": 1},
        }

        runner, results = run_plan([stage], CLIContext(instructions="hi"), expect_error=True)

        error = results["solo"].error
        assert "nonsense_option" in error
        assert "errors.pydantic.dev" not in error

    def test_cli_options_override_stage_options(self, tmp_path, fake_models, run_plan):
        stages = [
            {"name": "a", "summary": "s", "model": "echo", "prompt": "CLI",
             "options": {"temperature": 0.0, "max_tokens": 100}},
        ]
        cli = CLIContext(instructions="hi", options={"temperature": 0.7})

        runner, results = run_plan(stages, cli)

        options = fake_models.echo.prompts[0].options
        assert options.temperature == 0.7
        assert options.max_tokens == 100


class TestFailureHandling:
    def test_transient_failure_is_retried_then_succeeds(self, tmp_path, fake_models, run_plan):
        fake_models.flaky.failures_left = 2
        stages = [{"name": "solo", "summary": "s", "model": "flaky", "prompt": "CLI"}]

        runner, results = run_plan(stages, CLIContext(instructions="hi"))

        assert results["solo"].success
        assert fake_models.flaky.calls == 3

    def test_a_missing_api_key_fails_the_stage_without_retrying(self, tmp_path, fake_models, run_plan):
        stages = [{"name": "solo", "summary": "s", "model": "needskey", "prompt": "CLI"}]

        runner, results = run_plan(stages, CLIContext(instructions="hi"), expect_error=True)

        assert not results["solo"].success
        assert "key" in results["solo"].error.lower()
        assert fake_models.needskey.calls == 1

    @pytest.mark.parametrize("exception", [
        ValueError("unsupported attachment type"),
        NotImplementedError("this model cannot use tools"),
        TypeError("unexpected keyword argument"),
    ], ids=["ValueError", "NotImplementedError", "TypeError"])
    def test_a_validation_style_error_fails_the_stage_without_retrying(
        self, tmp_path, fake_models, run_plan, exception
    ):
        fake_models.raising.exception = exception

        stages = [{"name": "solo", "summary": "s", "model": "raising", "prompt": "CLI"}]
        runner, results = run_plan(stages, CLIContext(instructions="hi"), expect_error=True)

        assert not results["solo"].success
        assert str(exception) in results["solo"].error
        assert fake_models.raising.calls == 1

    def test_failure_after_retries_exhausted_fails_the_stage(self, tmp_path, fake_models, run_plan):
        fake_models.flaky.failures_left = 10
        stages = [{"name": "solo", "summary": "s", "model": "flaky", "prompt": "CLI"}]

        runner, results = run_plan(stages, CLIContext(instructions="hi"), expect_error=True)

        assert not results["solo"].success
        assert "transient upstream error" in results["solo"].error
        assert fake_models.flaky.calls == 3

    def test_intolerant_dependents_after_a_tolerated_failure_are_skipped(
        self, tmp_path, fake_models
    , run_plan):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "rescue", "summary": "s", "model": "echo", "prompt": "CLI",
             "depends_on": ["bad"], "partial_dependencies": True},
            {"name": "after", "summary": "s", "model": "echo",
             "depends_on": ["bad"], "prompt": "inline:Continue."},
        ]

        runner, results = run_plan(stages, CLIContext(instructions="hi"))

        assert results["rescue"].success
        assert not results["after"].success
        assert "bad" in results["after"].error

    @pytest.mark.parametrize("max_workers", [1, 4], ids=["sequential", "parallel"])
    def test_partial_dependencies_joins_only_surviving_outputs(
        self, tmp_path, fake_models, max_workers
    , run_plan):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "good", "summary": "s", "model": "echo", "prompt": "CLI",
             "produces": "Good Analysis"},
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "synthesis", "summary": "s", "model": "echo",
             "depends_on": ["bad", "good"], "partial_dependencies": True,
             "prompt": "inline:Synthesise."},
        ]

        runner, results = run_plan(
            stages, CLIContext(instructions="hi"), max_workers=max_workers
        )

        assert results["synthesis"].success
        assert "## Good Analysis" in results["synthesis"].text
        assert "## Output from bad" not in results["synthesis"].text

    @pytest.mark.parametrize("max_workers", [1, 4], ids=["sequential", "parallel"])
    def test_continue_on_failure_dependents_run_after_a_failed_dependency(
        self, tmp_path, fake_models, max_workers
    , run_plan):
        fake_models.flaky.failures_left = 10
        stages = [
            {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
            {"name": "rescue", "summary": "s", "model": "echo", "prompt": "CLI",
             "depends_on": ["bad"], "continue_on_failure": True},
        ]

        runner, results = run_plan(
            stages, CLIContext(instructions="hi"), max_workers=max_workers
        )

        assert not results["bad"].success
        assert results["rescue"].success
        assert results["rescue"].text == "ECHO[hi]"

    def test_llm_raise_errors_env_reraises_the_original_exception(
        self, tmp_path, fake_models, monkeypatch, run_plan
    ):
        monkeypatch.setenv("LLM_RAISE_ERRORS", "1")
        fake_models.flaky.failures_left = 10
        stages = [{"name": "solo", "summary": "s", "model": "flaky", "prompt": "CLI"}]

        with pytest.raises(RuntimeError, match="transient upstream error"):
            run_plan(stages, CLIContext(instructions="hi"))

        assert fake_models.flaky.calls == 1

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

    def test_load_plan_rejects_unknown_runtime_variables_in_env_values(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("", encoding="utf-8")
        plan_file = write_plan(
            tmp_path,
            [{"name": "a", "summary": "s", "type": "python_script", "script": "s.py",
              "env": {"CUSTOM_VAR": "${cli.bogus}"}}],
        )

        with pytest.raises(PlanError, match="cli.bogus"):
            load_plan(plan_file)

    @pytest.mark.parametrize("prompt", [
        "inline:Write results into ${run.output_dir}",
        [{"prompt": "inline:Run ${cli.run_id}", "label": "Task"}],
    ], ids=["scalar", "list"])
    def test_load_plan_rejects_runtime_tokens_in_llm_prompts(self, tmp_path, prompt):
        plan_file = write_plan(
            tmp_path,
            [{"name": "a", "summary": "s", "model": "echo", "prompt": prompt}],
        )

        with pytest.raises(PlanError, match="prompt"):
            load_plan(plan_file)
