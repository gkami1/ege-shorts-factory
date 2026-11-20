"""Rebuild a usable HF hub cache on local disk from a network volume that lost the snapshot symlinks.

RunPod network volumes (geesefs/S3) don't store symlinks: every hub/models--*/snapshots/<rev>/<file> comes
back as a 0-byte file after a restart, while the real data sits intact in blobs/. This script asks the Hub
which blob each file is (git blob id, or the LFS sha256) and recreates hub/ under DST with refs/ copied and
snapshots/ as real symlinks into the volume's blobs/. Then: export HF_HOME=<DST parent of hub>.

  python tools/hf_relink.py /workspace/cache/hf/hub /root/hf/hub
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi


def relink(src_hub: Path, dst_hub: Path) -> None:
    api = HfApi()
    for repo_dir in sorted(src_hub.glob("models--*")):
        repo_id = repo_dir.name[len("models--"):].replace("--", "/")
        dst = dst_hub / repo_dir.name
        if (dst / "refs").exists():
            shutil.rmtree(dst / "refs")
        shutil.copytree(repo_dir / "refs", dst / "refs")
        blobs = {p.name for p in (repo_dir / "blobs").iterdir()}
        for snap in (repo_dir / "snapshots").iterdir():
            info = api.model_info(repo_id, revision=snap.name, files_metadata=True)
            linked = missing = 0
            for f in info.siblings:
                blob = f.lfs.sha256 if f.lfs else f.blob_id
                if not (snap / f.rfilename).exists():
                    continue  # not downloaded on the volume: leave it out, HF downloads it on demand
                if blob not in blobs:
                    missing += 1
                    print(f"  ! {repo_id}: {f.rfilename} has no blob {blob}")
                    continue
                link = dst / "snapshots" / snap.name / f.rfilename
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(repo_dir / "blobs" / blob)
                linked += 1
            print(f"{repo_id}@{snap.name[:8]}: {linked} files linked, {missing} missing")


if __name__ == "__main__":
    relink(Path(sys.argv[1]), Path(sys.argv[2]))
