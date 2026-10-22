import requests

from pathlib import Path
from huggingface_hub import snapshot_download


DATA_DIR = Path(__file__).parent.resolve()

DATASETS = {
    "hatecheck": {
        "type": "huggingface",
        "repo_id": "Paul/hatecheck",
        "revision": "9d2ac89df04254e5c427bcc8d61b6d6c83a1f59b",
    },
    "multipico": {
        "type": "github",
        "repo": "Le-Wi-Di/le-wi-di.github.io",
        "revision": "52a2147263a244164fe1ba72fcd8d91c58375ebb",
        "path": "LeWiDi_3-2025/MP",
        "files": [
            "MP_annotators_meta.json",
            "MP_dev.json",
            "MP_test.json",
            "MP_train.json",
            "readme.md",
        ],
    },
}


def download_huggingface_dataset(name: str, repo_id: str, revision: str) -> None:
    output_dir = DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name}...")
    print(f"- Repository: {repo_id}")
    print(f"- Revision: {revision}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
    )

    print(f"- Saved to: {output_dir}")


def download_github_dataset(name: str, repo: str, revision: str, path: str, files: list[str],) -> None:
    output_dir = DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name}...")
    print(f"- Repository: {repo}")
    print(f"- Path: {path}")
    print(f"- Revision: {revision}")

    base_url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"

    for filename in files:
        url = f"{base_url}/{filename}"
        output_file = output_dir / filename

        print(f"- Downloading {filename}...")

        response = requests.get(url)
        response.raise_for_status()

        output_file.write_bytes(response.content)

    print(f"- Saved to: {output_dir}")


def download_dataset(name: str, config: dict) -> None:
    dataset_type = config["type"]

    if dataset_type == "huggingface":
        download_huggingface_dataset(
            name=name,
            repo_id=config["repo_id"],
            revision=config["revision"],
        )

    elif dataset_type == "github":
        download_github_dataset(
            name=name,
            repo=config["repo"],
            revision=config["revision"],
            path=config["path"],
            files=config["files"],
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == "__main__":
    for name, config in DATASETS.items():
        download_dataset(name, config)
        