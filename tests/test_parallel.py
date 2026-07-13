import time

import pytest
import yaml

from llm_plan.runner import CLIContext, PlanRunner, load_plan


def parallel_plan(tmp_path, stages, max_workers=4, **runner_kwargs):
    data = {
        "name": "test",
        "summary": "a parallel test plan",
        "parallel_config": {"max_workers": max_workers},
        "stages": stages,
    }
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return PlanRunner(
        load_plan(plan_file),
        CLIContext(instructions="go"),
        retry_delay=0,
        retries=0,
        **runner_kwargs,
    )


def by_name(results):
    return {r.name: r for r in results}


class TestParallelExecution:
    def test_independent_stages_run_concurrently(self, tmp_path):
        runner = parallel_plan(
            tmp_path,
            [
                {"name": "left", "summary": "s", "model": "pair", "prompt": "CLI"},
                {"name": "right", "summary": "s", "model": "pair", "prompt": "CLI"},
            ],
        )

        results = by_name(runner.run())

        assert results["left"].success and results["right"].success

    def test_diamond_joins_both_branch_outputs(self, tmp_path):
        runner = parallel_plan(
            tmp_path,
            [
                {"name": "root", "summary": "s", "model": "echo", "prompt": "CLI",
                 "produces": "Root"},
                {"name": "left", "summary": "s", "model": "echo", "depends_on": ["root"],
                 "prompt": "inline:left work", "produces": "Left View"},
                {"name": "right", "summary": "s", "model": "echo", "depends_on": ["root"],
                 "prompt": "inline:right work", "produces": "Right View"},
                {"name": "join", "summary": "s", "model": "echo",
                 "depends_on": ["left", "right"], "prompt": "inline:combine"},
            ],
        )

        results = by_name(runner.run())

        assert all(r.success for r in results.values())
        assert "## Left View" in results["join"].text
        assert "## Right View" in results["join"].text

    def test_results_are_returned_in_listed_stage_order(self, tmp_path):
        runner = parallel_plan(
            tmp_path,
            [
                {"name": "b_stage", "summary": "s", "model": "echo", "prompt": "CLI"},
                {"name": "a_stage", "summary": "s", "model": "echo", "prompt": "CLI"},
            ],
        )

        results = runner.run()

        assert [r.name for r in results] == ["b_stage", "a_stage"]


class TestExclusive:
    def test_exclusive_stage_never_overlaps_other_stages(self, tmp_path, fake_models):
        runner = parallel_plan(
            tmp_path,
            [
                {"name": "one", "summary": "s", "model": "timing",
                 "prompt": "inline:one", "depends_on": ["seed"]},
                {"name": "two", "summary": "s", "model": "timing",
                 "prompt": "inline:two", "depends_on": ["seed"]},
                {"name": "alone", "summary": "s", "model": "timing",
                 "prompt": "inline:alone", "depends_on": ["seed"], "exclusive": True},
                {"name": "after", "summary": "s", "model": "timing",
                 "prompt": "inline:after", "depends_on": ["seed"]},
            ],
        )

        results = by_name(runner.run())

        assert all(r.success for r in results.values())
        intervals = {text: (start, end) for text, start, end in fake_models.timing.intervals}
        assert len(intervals) == 4
        alone_start, alone_end = intervals.pop("alone")
        for text, (start, end) in intervals.items():
            assert end <= alone_start or start >= alone_end, (
                f"stage '{text}' ({start:.3f}-{end:.3f}) overlapped the exclusive "
                f"stage ({alone_start:.3f}-{alone_end:.3f})"
            )


class TestKeyboardInterrupt:
    def test_ctrl_c_abandons_in_flight_stages_and_never_starts_pending_ones(
        self, tmp_path, fake_models
    ):
        # The progress callback runs on the coordinating thread, so raising
        # from it lands the interrupt inside the scheduling loop while the
        # "first" stage is deterministically still in flight.
        def interrupt_at_second_stage(message):
            if message.startswith("Stage") and "second" in message:
                raise KeyboardInterrupt

        runner = parallel_plan(
            tmp_path,
            [
                {"name": "first", "summary": "s", "model": "hold", "prompt": "CLI"},
                {"name": "second", "summary": "s", "model": "echo", "prompt": "CLI"},
                {"name": "third", "summary": "s", "model": "echo", "prompt": "CLI"},
            ],
            progress=interrupt_at_second_stage,
        )

        start = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                runner.run()
        finally:
            fake_models.hold.release.set()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"ctrl-C blocked for {elapsed:.1f}s on in-flight stages"
        assert fake_models.echo.prompts == []
        assert "second" not in runner.results
        assert "third" not in runner.results


class TestParallelFailures:
    def test_llm_raise_errors_env_propagates_from_worker_threads(
        self, tmp_path, fake_models, monkeypatch
    ):
        monkeypatch.setenv("LLM_RAISE_ERRORS", "1")
        fake_models.flaky.failures_left = 10
        runner = parallel_plan(
            tmp_path,
            [
                {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
                {"name": "fine", "summary": "s", "model": "echo", "prompt": "CLI"},
            ],
        )

        with pytest.raises(RuntimeError, match="transient upstream error"):
            runner.run()

    def test_a_fatal_callback_error_cancels_queued_stages(self, tmp_path, fake_models):
        # Two workers, one fast stage and four held ones: when logging the
        # fast stage fails, the queued holds must be cancelled - draining
        # them would bill every queued stage and block for their duration.
        def failing_log(stage, response):
            raise RuntimeError("logs.db is broken")

        stages = [{"name": "fast", "summary": "s", "model": "echo", "prompt": "CLI"}] + [
            {"name": f"held{i}", "summary": "s", "model": "hold", "prompt": "CLI"}
            for i in range(4)
        ]
        runner = parallel_plan(tmp_path, stages, max_workers=2, on_response=failing_log)

        start = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="logs.db is broken"):
                runner.run()
        finally:
            fake_models.hold.release.set()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"fatal error blocked {elapsed:.1f}s draining the queue"
        # At most the in-flight holds ever started; the queued ones never ran.
        assert fake_models.hold.calls <= 2

    def test_failed_dependency_skips_intolerant_dependents(self, tmp_path, fake_models):
        fake_models.flaky.failures_left = 99
        runner = parallel_plan(
            tmp_path,
            [
                {"name": "bad", "summary": "s", "model": "flaky", "prompt": "CLI"},
                {"name": "child", "summary": "s", "model": "echo",
                 "depends_on": ["bad"], "prompt": "inline:child"},
                {"name": "grandchild", "summary": "s", "model": "echo",
                 "depends_on": ["child"], "prompt": "inline:grandchild"},
                {"name": "independent", "summary": "s", "model": "echo", "prompt": "CLI"},
            ],
        )

        results = by_name(runner.run())

        assert not results["bad"].success
        assert "skipped" in results["child"].error
        assert "skipped" in results["grandchild"].error
        assert results["independent"].success
