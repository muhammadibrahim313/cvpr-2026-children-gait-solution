from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    path = ROOT / script
    print(f"\n>>> running {path}")
    exec(path.read_text(encoding="utf-8"), {"__name__": "__main__"})


if __name__ == "__main__":
    run("stage3_feature_ensemble.py")

