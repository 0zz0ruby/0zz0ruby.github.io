"""Run all MA304 credit card project scripts in order."""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Run preprocessing, modeling, validation, and simulation scripts sequentially."""
    scripts = [
        SCRIPTS_DIR / "01_data_preprocessing.py",
        SCRIPTS_DIR / "02_pca_factor_clustering.py",
        SCRIPTS_DIR / "03_validation_visualization.py",
        # Optional report extension: Section 3 Simulation studies.
        # Comment out the next line if only the empirical Kaggle analysis is needed.
        SCRIPTS_DIR / "04_simulation_study.py",
    ]

    for script in scripts:
        print(f"Running {script.name}...")
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)

    print("All scripts completed successfully.")


if __name__ == "__main__":
    main()
