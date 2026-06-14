import os
from pathlib import Path

import requests
from huggingface_hub import get_token, hf_hub_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = PROJECT_ROOT / "external_datasets" / "gigaspeech_eval_parquet"
filename = "parquet-data/xs/validation-00000-of-00001.parquet"
target = LOCAL_DIR / filename
target.parent.mkdir(parents=True, exist_ok=True)
partial = target.with_suffix(target.suffix + ".part")
url = hf_hub_url("speechcolab/gigaspeech", filename, repo_type="dataset")
token = get_token()
if not token:
    raise RuntimeError(
        "GigaSpeech requires Hugging Face authentication. Run `hf auth login` first."
    )

if not target.exists():
    with requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(30, 300),
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(partial, target)

print(f"GigaSpeech validation ready: {target}")
print(f"Size: {target.stat().st_size / (1024 ** 3):.2f} GB")
