"""
Synthetic dataset generation with realistic patterns for software project time estimation.
"""

import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for project time estimation."""
    rng = np.random.RandomState(random_state)

    # --- Feature generation ---
    num_employees       = rng.randint(1, 51, n_samples)                      # 1-50
    complexity          = rng.randint(1, 11, n_samples)                      # 1-10
    team_experience     = rng.randint(1, 11, n_samples)                      # 1-10 (years avg)
    num_modules         = rng.randint(1, 31, n_samples)                      # 1-30
    working_hours_day   = rng.randint(4, 13, n_samples)                      # 4-12
    deadline_pressure   = rng.randint(1, 11, n_samples)                      # 1-10 (high=more pressure)
    requirement_clarity = rng.randint(1, 11, n_samples)                      # 1-10 (high=clear)

    # --- Base time formula (days) ---
    base_time = (
        (complexity * 8)
        + (num_modules * 3)
        - (team_experience * 1.5)
        - (num_employees * 0.8)
        + (deadline_pressure * 2)
        - (requirement_clarity * 1.2)
        - (working_hours_day * 1.0)
        + 30
    )

    # --- Interaction effects ---
    # Low clarity amplifies complexity impact
    base_time += (complexity * (10 - requirement_clarity)) * 0.3
    # Small team on complex project -> more time
    base_time += np.where(num_employees < 5, complexity * 2, 0)
    # Experienced team reduces time significantly
    base_time -= team_experience * num_employees * 0.05
    # Many modules with low experience -> overhead
    base_time += (num_modules * (10 - team_experience)) * 0.1

    # --- Noise ---
    noise = rng.normal(0, 5, n_samples)
    base_time = np.clip(base_time + noise, 10, 500)  # min 10 days, max 500 days

    df = pd.DataFrame({
        "num_employees":       num_employees,
        "complexity":          complexity,
        "team_experience":     team_experience,
        "num_modules":         num_modules,
        "working_hours_day":   working_hours_day,
        "deadline_pressure":   deadline_pressure,
        "requirement_clarity": requirement_clarity,
        "completion_days":     base_time.round(1),
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/projects.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df.describe())
