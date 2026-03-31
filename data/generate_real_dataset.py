import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="postgres",
    port=5433,
    password="pass"
)

query = """
    SELECT
        l.distance_km,
        l.speed_kmh,
        l.hour,
        l.weekday,
        p.delay_seconds,

        -- Both system predictions
        l.predicted_eta_seconds      AS l_predicted_eta_seconds,
        p.predicted_eta_seconds      AS p_predicted_eta_seconds,

        -- Engineered: real time elapsed between log creation and stop reached
        EXTRACT(EPOCH FROM (p.reached_at - l.created_at)) AS actual_travel_time_seconds,

        -- Target
        p.actual_eta_seconds

    FROM
        trip_eta_logs l
    JOIN
        trip_stop_progress p
    ON
        l.trip_id = p.trip_id
    AND
        l.next_stop_id = p.stop_id
    WHERE
        p.actual_eta_seconds IS NOT NULL
    AND
        p.reached_at IS NOT NULL
"""

print("Fetching data from PostgreSQL...")
df = pd.read_sql_query(query, conn)
conn.close()

# ── Basic cleaning ──────────────────────────────────────────────
df = df.dropna(subset=["distance_km","speed_kmh","hour","weekday","actual_eta_seconds"])
df = df[df["distance_km"].between(0.1, 100)]
df = df[df["speed_kmh"].between(2, 120)]
df = df[df["actual_eta_seconds"].between(30, 7200)]
df = df[df["delay_seconds"].between(-300, 3600)]
df = df[df["actual_travel_time_seconds"] > 0]

# ── Engineer new feature: how wrong was the old prediction? ─────
df["prediction_error"] = df["l_predicted_eta_seconds"] - df["actual_eta_seconds"]

df.to_csv("data/real_training_data.csv", index=False)
print(f"Dataset saved — shape: {df.shape}")
print(f"Columns: {list(df.columns)}")