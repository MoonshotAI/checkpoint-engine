import runpy
from pathlib import Path
from unittest.mock import patch


split_checkpoint_files = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "update.py"))[
    "split_checkpoint_files"
]


def test_split_checkpoint_files_is_deterministic_across_directory_order(tmp_path: Path) -> None:
    filenames = [
        "model-00003.safetensors",
        "notes.txt",
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]

    with patch("os.listdir", return_value=filenames):
        forward = [split_checkpoint_files(str(tmp_path), rank, 2) for rank in range(2)]

    with patch("os.listdir", return_value=list(reversed(filenames))):
        reverse = [split_checkpoint_files(str(tmp_path), rank, 2) for rank in range(2)]

    assert forward == reverse
    assert [path for shard in forward for path in shard] == [
        str(tmp_path / "model-00001.safetensors"),
        str(tmp_path / "model-00002.safetensors"),
        str(tmp_path / "model-00003.safetensors"),
    ]
