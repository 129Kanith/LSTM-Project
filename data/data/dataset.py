import pandas as pd
import random

TOTAL_ROWS = 15000

def generate_login_row():

    duration = random.randint(1, 20)

    protocol_type = random.choice(["tcp", "udp"])

    src_bytes = random.randint(50, 500)

    dst_bytes = random.randint(100, 2000)

    failed_logins = random.randint(0, 5)

    # login success logic
    if failed_logins > 3:
        logged_in = 0
    else:
        logged_in = random.choice([0, 1])

    count = random.randint(1, 50)

    srv_count = random.randint(1, 30)

    serror_rate = round(random.uniform(0.0, 0.5), 2)

    srv_serror_rate = round(random.uniform(0.0, 0.5), 2)

    return [
        duration,
        protocol_type,
        src_bytes,
        dst_bytes,
        failed_logins,
        logged_in,
        count,
        srv_count,
        serror_rate,
        srv_serror_rate
    ]


rows = []

print("Generating login activity dataset...")

for _ in range(TOTAL_ROWS):
    rows.append(generate_login_row())


columns = [
    "duration",
    "protocol_type",
    "src_bytes",
    "dst_bytes",
    "failed_logins",
    "logged_in",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate"
]

df = pd.DataFrame(rows, columns=columns)

# save dataset
df.to_csv(r"K:\LSTM\data\data\generated.csv", index=False)

print(f"Dataset generated successfully with {len(df)} records → generated.csv")