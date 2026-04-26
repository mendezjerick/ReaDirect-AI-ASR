from pathlib import Path

from readirect_asr.dataset.registry import list_active_datasets, load_dataset_registry


def test_dataset_registry_statuses(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "datasets:\n"
        "  readirect_content_bank:\n"
        "    status: active\n"
        "  cmudict:\n"
        "    status: active\n"
        "  speechocean762:\n"
        "    status: active\n"
        "  l2_arctic:\n"
        "    status: research_only_optional_noncommercial\n"
        "  pf_star:\n"
        "    status: future_optional_only\n",
        encoding="utf-8",
    )
    registry = load_dataset_registry(registry_path)
    assert registry["datasets"]["l2_arctic"]["status"] == "research_only_optional_noncommercial"
    assert registry["datasets"]["pf_star"]["status"] == "future_optional_only"
    assert set(list_active_datasets(registry)) == {"readirect_content_bank", "cmudict", "speechocean762"}
    assert "l2_arctic" not in list_active_datasets(registry)
    assert "pf_star" not in list_active_datasets(registry)
