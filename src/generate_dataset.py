from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_RECORDS = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "employee_burnout_synthetic.csv"


def _clip_round(values, lower, upper, decimals=0):
    values = np.clip(values, lower, upper)
    return np.round(values, decimals)


def generate_dataset(n_records=N_RECORDS, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    departments = np.array(["IT", "Sales", "HR", "Finance", "Operations", "Marketing"])
    roles = np.array(["Junior", "Mid", "Senior", "Lead", "Manager"])

    department = rng.choice(
        departments,
        size=n_records,
        p=[0.26, 0.19, 0.10, 0.14, 0.20, 0.11],
    )
    role = rng.choice(roles, size=n_records, p=[0.23, 0.31, 0.24, 0.13, 0.09])

    seniority_years = _clip_round(rng.gamma(shape=2.0, scale=2.1, size=n_records), 0, 20, 1)

    dept_overtime_effect = {
        "IT": 3.5,
        "Sales": 2.5,
        "HR": -1.0,
        "Finance": 1.5,
        "Operations": 2.0,
        "Marketing": 1.0,
    }
    role_overtime_effect = {
        "Junior": -1.0,
        "Mid": 0.0,
        "Senior": 1.5,
        "Lead": 3.0,
        "Manager": 4.0,
    }

    overtime_base = rng.normal(5.5, 4.0, n_records)
    overtime_hours = overtime_base + np.vectorize(dept_overtime_effect.get)(department)
    overtime_hours += np.vectorize(role_overtime_effect.get)(role)
    overtime_hours = _clip_round(overtime_hours, 0, 25, 1)

    total_weekly_hours = _clip_round(
        rng.normal(39.5, 3.0, n_records) + overtime_hours * 0.82,
        32,
        70,
        1,
    )

    role_meeting_effect = {
        "Junior": -0.4,
        "Mid": 0.0,
        "Senior": 0.4,
        "Lead": 0.9,
        "Manager": 1.2,
    }
    daily_meeting_volume = rng.normal(2.1, 0.8, n_records)
    daily_meeting_volume += np.vectorize(role_meeting_effect.get)(role)
    daily_meeting_volume += np.where(department == "Sales", 0.45, 0.0)
    daily_meeting_volume = _clip_round(daily_meeting_volume, 0, 7, 1)

    days_since_last_vacation = rng.gamma(shape=3.0, scale=38.0, size=n_records)
    days_since_last_vacation += overtime_hours * rng.normal(3.0, 0.55, n_records)
    days_since_last_vacation = _clip_round(days_since_last_vacation, 2, 420, 0).astype(int)

    sick_leave_days_6m = rng.poisson(
        lam=np.clip(1.1 + overtime_hours / 12 + days_since_last_vacation / 210, 0.2, 8.0),
        size=n_records,
    )
    sick_leave_days_6m = np.clip(sick_leave_days_6m, 0, 22)

    stress_latent = (
        1.55
        + overtime_hours / 8.0
        + daily_meeting_volume / 3.8
        + days_since_last_vacation / 230.0
        + rng.normal(0, 0.55, n_records)
    )
    self_reported_stress = np.clip(np.rint(stress_latent), 1, 5).astype(int)

    motivation_latent = (
        4.75
        - overtime_hours / 9.5
        - days_since_last_vacation / 260.0
        - sick_leave_days_6m / 14.0
        + rng.normal(0, 0.6, n_records)
    )
    motivation_level = np.clip(np.rint(motivation_latent), 1, 5).astype(int)

    risk_score = (
        0.19 * overtime_hours
        + 0.018 * days_since_last_vacation
        + 0.45 * daily_meeting_volume
        + 0.42 * sick_leave_days_6m
        + 1.15 * self_reported_stress
        - 0.95 * motivation_level
        + 0.08 * total_weekly_hours
        + np.where(department == "Sales", 0.35, 0.0)
        + np.where(department == "IT", 0.25, 0.0)
        + np.where(role == "Manager", 0.35, 0.0)
        + rng.normal(0, 1.1, n_records)
    )

    low_threshold = np.quantile(risk_score, 0.37)
    high_threshold = np.quantile(risk_score, 0.74)
    burnout_risk = np.where(
        risk_score >= high_threshold,
        "High",
        np.where(risk_score >= low_threshold, "Medium", "Low"),
    )

    employee_id = [f"E{idx:04d}" for idx in range(1, n_records + 1)]
    dataset = pd.DataFrame(
        {
            "employee_id": employee_id,
            "department": department,
            "role": role,
            "seniority_years": seniority_years,
            "total_weekly_hours": total_weekly_hours,
            "overtime_hours": overtime_hours,
            "daily_meeting_volume": daily_meeting_volume,
            "days_since_last_vacation": days_since_last_vacation,
            "sick_leave_days_6m": sick_leave_days_6m,
            "self_reported_stress": self_reported_stress,
            "motivation_level": motivation_level,
            "burnout_risk": burnout_risk,
        }
    )

    return dataset


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    print(f"Generated {len(dataset)} records at {DATASET_PATH}")
    print(dataset["burnout_risk"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
