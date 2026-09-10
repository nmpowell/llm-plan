import io

from llm_plan.script_stage import (
    emit_outputs,
    manifest_label,
    manifest_output_paths,
    parse_manifest,
)


class TestManifestRoundTrip:
    def test_path_label_kind_and_metadata_survive_emit_and_parse(self, tmp_path):
        stream = io.StringIO()
        emit_outputs(
            [
                {"path": tmp_path / "report.md", "label": "Report", "kind": "output"},
                str(tmp_path / "raw.json"),
            ],
            metadata={"model": "haiku", "cost_usd": 0.01},
            stream=stream,
        )

        manifest = parse_manifest(stream.getvalue())

        assert manifest["outputs"][0] == {
            "path": str(tmp_path / "report.md"),
            "label": "Report",
            "kind": "output",
        }
        assert manifest["outputs"][1] == {"path": str(tmp_path / "raw.json")}
        assert manifest["metadata"] == {"model": "haiku", "cost_usd": 0.01}
        assert manifest_output_paths(manifest) == [
            tmp_path / "report.md",
            tmp_path / "raw.json",
        ]
        assert manifest_label(manifest, tmp_path / "report.md") == "Report"

    def test_bare_lines_mode_emits_paths_only(self, tmp_path):
        stream = io.StringIO()

        emit_outputs(
            [tmp_path / "a.md", tmp_path / "b.md"], manifest=False, stream=stream
        )

        assert stream.getvalue().splitlines() == [
            str(tmp_path / "a.md"),
            str(tmp_path / "b.md"),
        ]
        assert parse_manifest(stream.getvalue()) is None
