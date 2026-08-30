from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from app.model import BIREFNET_REPOSITORY, BIREFNET_REVISION

QUALITY_MODEL_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
)


def prefetch_quality_model() -> str:
    arguments = {
        "repo_id": BIREFNET_REPOSITORY,
        "revision": BIREFNET_REVISION,
        "allow_patterns": QUALITY_MODEL_FILES,
    }
    try:
        return snapshot_download(**arguments, local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(**arguments)


def main() -> None:
    path = prefetch_quality_model()
    print(f"BiRefNet cached at: {path}")


if __name__ == "__main__":
    main()
