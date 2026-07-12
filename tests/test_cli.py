from click.testing import CliRunner
from llm.cli import cli


class TestPlanPath:
    def test_prints_the_user_plans_directory(self, user_dir):
        runner = CliRunner()

        result = runner.invoke(cli, ["plan", "path"])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == str(user_dir / "plans")
