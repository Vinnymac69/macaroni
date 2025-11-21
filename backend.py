from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Literal

# ---------- Paths & Setup ----------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SESSIONS_PATH = DATA_DIR / "sessions.csv"
EXERCISES_PATH = DATA_DIR / "exercises.csv"
SETS_PATH = DATA_DIR / "sets.csv"

# ---------- Load / Save ----------

def _empty_sessions_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["session_id", "date", "group", "notes"]
    )

def _empty_exercises_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["exercise_id", "session_id", "exercise_name", "muscle", "group"]
    )

def _empty_sets_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["set_id", "exercise_id", "set_number", "weight", "reps", "rir", "comments"]
    )

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three tables, creating empty ones if files don’t exist yet."""
    if SESSIONS_PATH.exists():
        sessions = pd.read_csv(SESSIONS_PATH, parse_dates=["date"])
    else:
        sessions = _empty_sessions_df()

    if EXERCISES_PATH.exists():
        exercises = pd.read_csv(EXERCISES_PATH)
    else:
        exercises = _empty_exercises_df()

    if SETS_PATH.exists():
        sets = pd.read_csv(SETS_PATH)
    else:
        sets = _empty_sets_df()

    return sessions, exercises, sets

def save_data(sessions: pd.DataFrame, exercises: pd.DataFrame, sets: pd.DataFrame) -> None:
    """Persist tables to CSV."""
    sessions.to_csv(SESSIONS_PATH, index=False)
    exercises.to_csv(EXERCISES_PATH, index=False)
    sets.to_csv(SETS_PATH, index=False)

# ---------- ID helpers ----------

def _next_id(df: pd.DataFrame, id_col: str) -> int:
    if df.empty:
        return 1
    return int(df[id_col].max()) + 1

# ---------- Create / log functions ----------

def add_session(
    sessions: pd.DataFrame,
    date,
    group: str,
    notes: str = ""
) -> Tuple[pd.DataFrame, int]:
    """
    Add a new training session (e.g. Push day).
    Returns updated sessions df and the new session_id.
    """
    session_id = _next_id(sessions, "session_id")
    new_row = {
        "session_id": session_id,
        "date": pd.to_datetime(date),
        "group": group,
        "notes": notes,
    }
    sessions = pd.concat([sessions, pd.DataFrame([new_row])], ignore_index=True)
    return sessions, session_id

def add_exercise(
    exercises: pd.DataFrame,
    session_id: int,
    exercise_name: str,
    muscle: Optional[str] = None,
    group: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Add an exercise performed within a session.
    e.g. Bench Press in a Push session.
    """
    exercise_id = _next_id(exercises, "exercise_id")
    new_row = {
        "exercise_id": exercise_id,
        "session_id": session_id,
        "exercise_name": exercise_name,
        "muscle": muscle if muscle is not None else "",
        "group": group if group is not None else "",
    }
    exercises = pd.concat([exercises, pd.DataFrame([new_row])], ignore_index=True)
    return exercises, exercise_id

def add_set(
    sets: pd.DataFrame,
    exercise_id: int,
    set_number: int,
    weight: float,
    reps: int,
    rir: Optional[int] = None,
    comments: str = "",
) -> pd.DataFrame:
    """
    Log a single set for an exercise.
    """
    set_id = _next_id(sets, "set_id")
    new_row = {
        "set_id": set_id,
        "exercise_id": exercise_id,
        "set_number": set_number,
        "weight": float(weight),
        "reps": int(reps),
        "rir": int(rir) if rir is not None else None,
        "comments": comments,
    }
    sets = pd.concat([sets, pd.DataFrame([new_row])], ignore_index=True)
    return sets

def quick_log_set(
    sessions: pd.DataFrame,
    exercises: pd.DataFrame,
    sets: pd.DataFrame,
    date,
    day_group: str,
    exercise_name: str,
    set_number: int,
    weight: float,
    reps: int,
    rir: Optional[int] = None,
    muscle: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to log a set in one call.
    - Creates session if no session for that date+group exists yet.
    - Creates exercise row under that session.
    - Appends the set.
    """
    # Ensure session exists for date+group
    date = pd.to_datetime(date)
    mask = (sessions["date"] == date) & (sessions["group"] == day_group)
    if mask.any():
        session_id = int(sessions.loc[mask, "session_id"].iloc[0])
    else:
        sessions, session_id = add_session(sessions, date, day_group)

    # Ensure exercise exists under that session
    mask_ex = (exercises["session_id"] == session_id) & (exercises["exercise_name"] == exercise_name)
    if mask_ex.any():
        exercise_id = int(exercises.loc[mask_ex, "exercise_id"].iloc[0])
    else:
        exercises, exercise_id = add_exercise(
            exercises,
            session_id=session_id,
            exercise_name=exercise_name,
            muscle=muscle,
            group=day_group,
        )

    # Add the set
    sets = add_set(
        sets,
        exercise_id=exercise_id,
        set_number=set_number,
        weight=weight,
        reps=reps,
        rir=rir,
    )

    return sessions, exercises, sets

# ---------- Analytics helpers ----------

def _joined_sets(
    sessions: pd.DataFrame,
    exercises: pd.DataFrame,
    sets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join sets + exercises + sessions into one flat table.
    """
    df = sets.merge(exercises, on="exercise_id", how="left", suffixes=("", "_ex"))
    df = df.merge(sessions, on="session_id", how="left", suffixes=("", "_sess"))
    # Normalize column names:
    df = df.rename(columns={"date": "session_date", "group": "session_group"})
    return df

def exercise_history(
    sessions: pd.DataFrame,
    exercises: pd.DataFrame,
    sets: pd.DataFrame,
    exercise_name: str,
) -> pd.DataFrame:
    """
    Return all sets for a given exercise across time.
    Columns: session_date, set_number, weight, reps, rir, session_group, notes, etc.
    """
    df = _joined_sets(sessions, exercises, sets)
    mask = df["exercise_name"] == exercise_name
    return df.loc[mask].sort_values(["session_date", "set_number"])

def exercise_progress(
    sessions: pd.DataFrame,
    exercises: pd.DataFrame,
    sets: pd.DataFrame,
    exercise_name: str,
    metric: Literal["top_set_weight", "total_volume", "e1rm"] = "top_set_weight",
) -> pd.DataFrame:
    """
    Aggregate per session for a given exercise:
    - 'top_set_weight': max weight per session
    - 'total_volume': sum(weight * reps) per session
    - 'e1rm': estimated 1RM (max across sets per session)
    """
    df = exercise_history(sessions, exercises, sets, exercise_name)
    if df.empty:
        return df

    if metric == "top_set_weight":
        agg = df.groupby("session_date")["weight"].max().reset_index(name="value")
    elif metric == "total_volume":
        df["volume"] = df["weight"] * df["reps"]
        agg = df.groupby("session_date")["volume"].sum().reset_index(name="value")
    elif metric == "e1rm":
        df["e1rm"] = df["weight"] * (1 + df["reps"] / 30.0)
        agg = df.groupby("session_date")["e1rm"].max().reset_index(name="value")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return agg.sort_values("session_date")

def add_week_columns(sessions: pd.DataFrame) -> pd.DataFrame:
    """Add ISO week/year columns to sessions (for weekly stats)."""
    s = sessions.copy()
    s["week"] = s["date"].dt.isocalendar().week
    s["year"] = s["date"].dt.year
    return s

def weekly_volume_by_muscle(
    sessions: pd.DataFrame,
    exercises: pd.DataFrame,
    sets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weekly set counts and volume by muscle group.
    Assumes 'muscle' column in exercises.
    """
    s = add_week_columns(sessions)
    df = _joined_sets(s, exercises, sets)
    if df.empty:
        return df

    df["week"] = df["session_date"].dt.isocalendar().week
    df["year"] = df["session_date"].dt.year
    df["volume"] = df["weight"] * df["reps"]

    weekly = (
        df.groupby(["year", "week", "muscle"])
          .agg(
              sets=("set_id", "count"),
              volume=("volume", "sum"),
          )
          .reset_index()
    )
    return weekly
