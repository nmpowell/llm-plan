import pytest
import yaml

from llm_plan.models import PlanError, PromptType
from llm_plan.parser import (
    load_with_extends,
    parse_plan,
    parse_prompt_field,
    parse_prompt_list,
    prompt_has_cli,
    substitute_variables,
)


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def minimal_plan(**overrides):
    data = {
        "name": "test",
        "summary": "a test plan",
        "stages": [
            {"name": "first", "summary": "first stage", "model": "echo", "prompt": "CLI"},
        ],
    }
    data.update(overrides)
    return data


class TestLoadWithExtends:
    def test_merges_defaults_underneath_the_plan(self, tmp_path):
        write_yaml(tmp_path / "_defaults.yaml", {"models": {"fast": "echo", "slow": "opus"}})
        plan_file = write_yaml(
            tmp_path / "plan.yaml",
            {"extends": "_defaults.yaml", "models": {"slow": "sonnet"}, "name": "p"},
        )

        data = load_with_extends(plan_file)

        assert data["models"] == {"fast": "echo", "slow": "sonnet"}
        assert data["name"] == "p"
        assert "extends" not in data

    def test_extends_resolves_recursively_through_a_grandparent(self, tmp_path):
        write_yaml(
            tmp_path / "_grand.yaml", {"models": {"fast": "echo", "slow": "opus"}}
        )
        write_yaml(
            tmp_path / "_base.yaml",
            {"extends": "_grand.yaml", "models": {"slow": "sonnet"}},
        )
        plan_file = write_yaml(
            tmp_path / "plan.yaml", {"extends": "_base.yaml", "name": "p"}
        )

        data = load_with_extends(plan_file)

        assert data["models"] == {"fast": "echo", "slow": "sonnet"}
        assert "extends" not in data

    def test_circular_extends_is_an_error_naming_the_files(self, tmp_path):
        write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml", "name": "a"})
        plan_file = write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml", "name": "b"})

        with pytest.raises(PlanError, match=r"[Cc]ircular.*b\.yaml.*a\.yaml.*b\.yaml"):
            load_with_extends(plan_file)

    def test_missing_extends_target_is_an_error(self, tmp_path):
        plan_file = write_yaml(tmp_path / "plan.yaml", {"extends": "nope.yaml"})

        with pytest.raises(PlanError, match="nope.yaml"):
            load_with_extends(plan_file)


class TestSubstituteVariables:
    def test_substitutes_nested_paths(self):
        result = substitute_variables(
            {"model": "${models.opus}"}, {"models": {"opus": "claude-opus"}}
        )

        assert result == {"model": "claude-opus"}

    def test_unknown_variable_is_an_error_naming_available_keys(self):
        with pytest.raises(PlanError, match=r"\$\{models\.nope\}.*models"):
            substitute_variables("${models.nope}", {"models": {"opus": "x"}})

    def test_deferred_namespaces_are_left_verbatim(self):
        result = substitute_variables(
            "${cli.instructions} and ${models.opus}",
            {"models": {"opus": "claude-opus"}},
            defer=frozenset({"cli", "run"}),
        )

        assert result == "${cli.instructions} and claude-opus"


class TestParsePromptField:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("CLI", (PromptType.CLI, "")),
            ("cli", (PromptType.CLI, "")),
            ("CLI:instructions", (PromptType.CLI, "instructions")),
            ("CLI:files", (PromptType.CLI, "files")),
            ("CLI:all", (PromptType.CLI, "all")),
            ("inline:Focus on X", (PromptType.INLINE, "Focus on X")),
            ("chain:analyst", (PromptType.CHAIN, "analyst")),
            (None, (PromptType.NONE, "")),
        ],
    )
    def test_classifies_keywords(self, text, expected):
        assert parse_prompt_field(text, "stage") == expected

    def test_invalid_cli_suffix_is_an_error(self):
        with pytest.raises(PlanError, match="CLI:instructions, CLI:files, CLI:all"):
            parse_prompt_field("CLI:everything", "stage")

    def test_empty_chain_target_is_an_error(self):
        with pytest.raises(PlanError, match="chain:"):
            parse_prompt_field("chain:", "stage")

    def test_resolves_file_relative_to_the_plan_directory(self, tmp_path):
        (tmp_path / "prompt.md").write_text("do the thing", encoding="utf-8")

        ptype, value = parse_prompt_field("prompt.md", "stage", base_dir=tmp_path)

        assert ptype == PromptType.FILE
        assert value == str(tmp_path / "prompt.md")

    def test_missing_path_like_prompt_is_an_error_suggesting_inline(self, tmp_path):
        with pytest.raises(PlanError, match="inline:"):
            parse_prompt_field("prompts/missing.md", "stage", base_dir=tmp_path)

    def test_bare_words_become_inline_with_a_warning(self):
        with pytest.warns(UserWarning, match="inline"):
            ptype, value = parse_prompt_field("Summarise the input", "stage")

        assert (ptype, value) == (PromptType.INLINE, "Summarise the input")


class TestPromptHasCli:
    @pytest.mark.parametrize(
        "prompt, expected",
        [
            ("CLI", True),
            ("cli:files", True),
            ("inline:CLI", False),
            (["inline:x", "CLI:instructions"], True),
            ([{"prompt": "CLI", "label": "Task"}], True),
            ([{"prompt": "inline:x"}], False),
            (None, False),
        ],
    )
    def test_detects_cli_tokens_in_any_shape(self, prompt, expected):
        assert prompt_has_cli(prompt) is expected


class TestParsePromptList:
    def test_composes_files_inline_and_cli_sections_in_order(self, tmp_path):
        (tmp_path / "guide.md").write_text("guide text", encoding="utf-8")
        prompt = [
            {"prompt": "CLI:instructions", "label": "Original Question"},
            {"prompt": "guide.md", "label": "Guidelines"},
            "inline:Also consider Y",
        ]

        spec = parse_prompt_list(
            prompt, "stage", [(None, "the question")], completed_text={}, base_dir=tmp_path
        )

        assert [(f.label, f.path.name) for f in spec.prompt_files] == [("Guidelines", "guide.md")]
        assert spec.sections == [
            ("Original Question", "the question"),
            (None, "Also consider Y"),
        ]
        assert spec.wants_cli_files is False
        assert spec.requires_cli_content is False

    def test_chain_injects_completed_stage_text(self):
        spec = parse_prompt_list(
            [{"prompt": "chain:analyst", "label": "Analysis"}],
            "stage",
            [],
            completed_text={"analyst": "analyst says hi"},
        )

        assert spec.sections == [("Analysis", "analyst says hi")]

    def test_chain_to_unfinished_stage_is_an_error_listing_available(self):
        with pytest.raises(PlanError, match="analyst.*no completed output.*reviewer"):
            parse_prompt_list(
                "chain:analyst",
                "stage",
                [],
                completed_text={"reviewer": "text"},
            )

    def test_a_skipped_chain_target_is_omitted_not_an_error(self):
        spec = parse_prompt_list(
            ["chain:analyst", "inline:fallback"],
            "stage",
            [],
            completed_text={},
            skip_chain_targets=frozenset({"analyst"}),
        )

        assert spec.sections == [(None, "fallback")]

    def test_plain_cli_wants_files_and_requires_content(self):
        spec = parse_prompt_list("CLI", "stage", [], completed_text={})

        assert spec.wants_cli_files is True
        assert spec.requires_cli_content is True
        assert spec.sections == []

    def test_cli_instructions_variant_does_not_want_files(self):
        spec = parse_prompt_list(
            "CLI:instructions", "stage", [(None, "q")], completed_text={}
        )

        assert spec.wants_cli_files is False
        assert spec.sections == [(None, "q")]

    def test_yaml_inline_mapping_trap_gets_a_helpful_error(self):
        with pytest.raises(PlanError, match='"inline:'):
            parse_prompt_list(
                [{"inline": "some text"}], "stage", [], completed_text={}
            )


class TestParsePlan:
    def test_parses_a_minimal_plan(self, tmp_path):
        plan_file = write_yaml(tmp_path / "plan.yaml", minimal_plan())

        plan = parse_plan(plan_file)

        assert plan.name == "test"
        assert plan.max_workers == 1
        stage = plan.stages[0]
        assert (stage.name, stage.model, stage.type) == ("first", "echo", "llm")

    def test_substitutes_variables_from_any_top_level_namespace(self, tmp_path):
        data = minimal_plan(
            models={"fast": "echo"},
            strategies={"cot": "think step by step"},
        )
        data["stages"][0]["model"] = "${models.fast}"
        data["stages"][0]["prompt"] = "inline:${strategies.cot}"
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        plan = parse_plan(plan_file)

        assert plan.stages[0].model == "echo"
        assert plan.stages[0].prompt == "inline:think step by step"

    def test_undefined_variable_fails_at_load_even_without_namespaces(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["model"] = "${missing.value}"
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match=r"Variable '\$\{missing\.value\}' not found"):
            parse_plan(plan_file)

    def test_runtime_tokens_pass_through_untouched_without_namespaces(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["prompt"] = "inline:${cli.instructions} into ${run.output_dir}"
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        plan = parse_plan(plan_file)

        assert plan.stages[0].prompt == "inline:${cli.instructions} into ${run.output_dir}"

    def test_reads_parallel_config_max_workers(self, tmp_path):
        plan_file = write_yaml(
            tmp_path / "plan.yaml", minimal_plan(parallel_config={"max_workers": 7})
        )

        assert parse_plan(plan_file).max_workers == 7

    @pytest.mark.parametrize("bad_value", [0, -2, True, 2.9, "many"])
    def test_invalid_max_workers_is_an_error(self, tmp_path, bad_value):
        plan_file = write_yaml(
            tmp_path / "plan.yaml", minimal_plan(parallel_config={"max_workers": bad_value})
        )

        with pytest.raises(PlanError, match="max_workers"):
            parse_plan(plan_file)

    @pytest.mark.parametrize("bad_config", ["many", [1], 3, False])
    def test_non_mapping_parallel_config_is_an_error(self, tmp_path, bad_config):
        plan_file = write_yaml(
            tmp_path / "plan.yaml", minimal_plan(parallel_config=bad_config)
        )

        with pytest.raises(PlanError, match="parallel_config"):
            parse_plan(plan_file)

    def test_non_utf8_plan_file_is_a_plan_error(self, tmp_path):
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_bytes(b"name: \xff\xfe broken")

        with pytest.raises(PlanError, match="plan.yaml"):
            parse_plan(plan_file)

    def test_malformed_yaml_is_a_plan_error(self, tmp_path):
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("stages: [unclosed", encoding="utf-8")

        with pytest.raises(PlanError, match="plan.yaml"):
            parse_plan(plan_file)

    def test_non_mapping_document_is_a_plan_error(self, tmp_path):
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(PlanError, match="mapping"):
            parse_plan(plan_file)

    def test_non_mapping_stage_is_a_plan_error(self, tmp_path):
        plan_file = write_yaml(tmp_path / "plan.yaml", minimal_plan(stages=["oops"]))

        with pytest.raises(PlanError, match="[Ss]tage"):
            parse_plan(plan_file)

    @pytest.mark.parametrize(
        "mutation, message",
        [
            (lambda s: s.pop("name"), "'name' is required"),
            (lambda s: s.pop("summary"), "'summary' is required"),
            (lambda s: s.update(model=5), "'model' must be a non-empty string"),
            (lambda s: s.update(model=""), "'model' must be a non-empty string"),
            (lambda s: s.update(type="shell"), "shell"),
            (lambda s: s.update(timeout=-1), "positive integer"),
            (lambda s: s.update(timeout=True), "positive integer"),
            (lambda s: s.update(env=["a"]), "mapping"),
            (lambda s: s.update(options=["a"]), "mapping"),
            (lambda s: s.update(script_args="oops"), "list"),
            (lambda s: s.update(files=None), "'files' must be a list"),
            (lambda s: s.update(files="context.md"), "'files' must be a list"),
            (lambda s: s.update(attachments=42), "'attachments' must be a list"),
            (lambda s: s.update(attachments={}), "'attachments' must be a list"),
        ],
    )
    def test_stage_validation_errors(self, tmp_path, mutation, message):
        data = minimal_plan()
        mutation(data["stages"][0])
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match=message):
            parse_plan(plan_file)

    @pytest.mark.parametrize("bad_name", ["sub/dir", "back\\slash", 5, "", ".", ".."])
    def test_a_stage_name_must_be_a_plain_string(self, tmp_path, bad_name):
        # Stage names become scratch subdirectory names; a separator would
        # escape the run's scratch root.
        data = minimal_plan()
        data["stages"][0]["name"] = bad_name
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="name"):
            parse_plan(plan_file)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("depends_on", False),
            ("depends_on", 0),
            ("env", False),
            ("env", ""),
            ("options", 0),
            ("options", False),
            ("prompt", False),
            ("prompt", 0),
        ],
    )
    def test_falsey_malformed_fields_are_rejected_at_load(self, tmp_path, field, value):
        # `x or default` would silently turn these into empty defaults.
        data = minimal_plan()
        data["stages"][0][field] = value
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match=field):
            parse_plan(plan_file)

    @pytest.mark.parametrize("field", ["depends_on", "env", "options", "prompt"])
    def test_an_explicitly_empty_field_means_its_default(self, tmp_path, field):
        # YAML idiom: a key with no value (None) is fine, unlike a wrong type.
        data = minimal_plan()
        data["stages"][0][field] = None
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        plan = parse_plan(plan_file)

        assert plan.stages[0].name

    def test_an_llm_stage_without_a_model_uses_the_default(self, tmp_path):
        data = minimal_plan()
        data["stages"][0].pop("model")
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        plan = parse_plan(plan_file)

        assert plan.stages[0].model is None

    @pytest.mark.parametrize(
        "flag", ["exclusive", "continue_on_failure", "partial_dependencies"]
    )
    @pytest.mark.parametrize("bad_value", ["false", 0, 1, None], ids=repr)
    def test_boolean_stage_fields_reject_non_booleans(self, tmp_path, flag, bad_value):
        data = minimal_plan()
        data["stages"][0][flag] = bad_value
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match=f"first.*'{flag}' must be a boolean"):
            parse_plan(plan_file)

    @pytest.mark.parametrize("bad_value", ["opus", 3, {"stage": "opus"}], ids=repr)
    def test_non_list_depends_on_is_an_error(self, tmp_path, bad_value):
        data = minimal_plan()
        data["stages"][0]["depends_on"] = bad_value
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="'depends_on' must be a list"):
            parse_plan(plan_file)

    def test_non_string_depends_on_item_is_an_error(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["depends_on"] = ["opus", 3]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="'depends_on' entries must be strings"):
            parse_plan(plan_file)

    def test_duplicate_depends_on_entries_are_an_error(self, tmp_path):
        data = minimal_plan()
        data["stages"].append(
            {
                "name": "second",
                "summary": "second stage",
                "model": "echo",
                "prompt": "inline:go",
                "depends_on": ["first", "first"],
            }
        )
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="second.*duplicate.*first"):
            parse_plan(plan_file)

    def test_misspelt_stage_key_gets_a_did_you_mean(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["depend_on"] = ["other"]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(
            PlanError, match="first.*unknown key 'depend_on'.*did you mean 'depends_on'"
        ):
            parse_plan(plan_file)

    def test_unrecognisable_stage_key_lists_the_valid_keys(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["zzz_frobnicate"] = True
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(
            PlanError, match="unknown key 'zzz_frobnicate'.*Valid keys:.*depends_on"
        ):
            parse_plan(plan_file)

    def test_empty_plan_file_is_an_error(self, tmp_path):
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("", encoding="utf-8")

        with pytest.raises(PlanError, match="[Ee]mpty"):
            parse_plan(plan_file)

    def test_plan_without_stages_is_an_error(self, tmp_path):
        plan_file = write_yaml(tmp_path / "plan.yaml", {"name": "x", "summary": "y"})

        with pytest.raises(PlanError, match="[Ss]tages"):
            parse_plan(plan_file)

    def test_script_stage_resolves_script_relative_to_plan(self, tmp_path):
        script = tmp_path / "work.py"
        script.write_text("print('hi')", encoding="utf-8")
        data = minimal_plan()
        data["stages"] = [
            {
                "name": "script",
                "summary": "run a script",
                "type": "python_script",
                "script": "work.py",
                "script_args": ["--quiet"],
            }
        ]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        stage = parse_plan(plan_file).stages[0]

        assert stage.resolved_script == script
        assert stage.script_args == ["--quiet"]

    def test_missing_script_is_an_error(self, tmp_path):
        data = minimal_plan()
        data["stages"] = [
            {"name": "s", "summary": "x", "type": "python_script", "script": "gone.py"}
        ]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="gone.py"):
            parse_plan(plan_file)

    def test_stage_files_resolve_relative_to_plan_and_must_exist(self, tmp_path):
        (tmp_path / "context.md").write_text("ctx", encoding="utf-8")
        data = minimal_plan()
        data["stages"][0]["files"] = [{"path": "context.md", "label": "Context"}]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        stage = parse_plan(plan_file).stages[0]

        assert [(f.label, f.path) for f in stage.resolved_files] == [
            ("Context", tmp_path / "context.md")
        ]

    def test_missing_stage_file_is_an_error(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["files"] = [{"path": "gone.md"}]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="gone.md"):
            parse_plan(plan_file)

    @pytest.mark.parametrize(
        "bad_entry", ["context.md", {"label": "Context"}], ids=["bare-string", "no-path"]
    )
    def test_malformed_files_entry_is_a_plan_error(self, tmp_path, bad_entry):
        data = minimal_plan()
        data["stages"][0]["files"] = [bad_entry]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="files"):
            parse_plan(plan_file)

    @pytest.mark.parametrize(
        "bad_entry", ["img.png", {"label": "Diagram"}], ids=["bare-string", "no-path"]
    )
    def test_malformed_attachments_entry_is_a_plan_error(self, tmp_path, bad_entry):
        data = minimal_plan()
        data["stages"][0]["attachments"] = [bad_entry]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="attachments"):
            parse_plan(plan_file)

    def test_url_attachments_pass_through_and_paths_must_exist(self, tmp_path):
        (tmp_path / "img.png").write_bytes(b"\x89PNG")
        data = minimal_plan()
        data["stages"][0]["attachments"] = [
            {"path": "https://example.com/a.png", "label": "Remote"},
            {"path": "img.png"},
        ]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        stage = parse_plan(plan_file).stages[0]

        assert stage.resolved_attachments[0].uri == "https://example.com/a.png"
        assert stage.resolved_attachments[0].is_url
        assert stage.resolved_attachments[1].uri == str(tmp_path / "img.png")
        assert not stage.resolved_attachments[1].is_url

    def test_list_prompt_file_must_exist_at_load_time(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["prompt"] = [
            "CLI:instructions",
            {"prompt": "missing_prompt.md", "label": "Guide"},
        ]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="missing_prompt.md"):
            parse_plan(plan_file)

    def test_non_string_prompt_in_a_list_entry_is_an_error_at_load(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["prompt"] = [{"prompt": 123}]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="first.*123.*string"):
            parse_plan(plan_file)

    def test_non_string_prompt_label_is_an_error_at_load(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["prompt"] = [{"prompt": "inline:go", "label": 7}]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="first.*label.*string"):
            parse_plan(plan_file)

    def test_invalid_cli_form_in_a_list_entry_is_an_error_at_load(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["prompt"] = ["CLI:everything"]
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="CLI:instructions, CLI:files, CLI:all"):
            parse_plan(plan_file)

    def test_scalar_file_prompt_must_exist_at_load_time(self, tmp_path):
        data = minimal_plan()
        data["stages"][0]["prompt"] = "missing_prompt.md"
        plan_file = write_yaml(tmp_path / "plan.yaml", data)

        with pytest.raises(PlanError, match="missing_prompt.md"):
            parse_plan(plan_file)
