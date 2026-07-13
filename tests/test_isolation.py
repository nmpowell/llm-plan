import llm.plugins


class TestSuiteIsolation:
    def test_only_this_plugins_entry_point_is_loaded(self):
        llm.plugins.load_plugins()

        loaded = {name for name, _ in llm.plugins.pm.list_name_plugin()}

        assert "plan" in loaded
        unexpected = {
            name
            for name in loaded
            if name not in ("plan", "llm-plan-test-models")
            and not name.startswith("llm.default_plugins.")
        }
        assert unexpected == set(), (
            f"third-party plugins loaded into the test suite: {sorted(unexpected)}"
        )
