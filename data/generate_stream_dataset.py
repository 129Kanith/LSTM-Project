import pandas as pd
import numpy as np
import random

def generate_row(attack):
    if attack == 0:  # Normal
        # Typical traffic, low error rates, successful logins
        return [
            random.randint(1, 10),     # duration
            random.choice(["tcp", "udp", "icmp"]), # protocol_type
            random.randint(200, 1000), # src_bytes
            random.randint(1500, 5000),# dst_bytes
            0,                         # failed_logins (zero for normal)
            1,                         # logged_in (usually logged in)
            random.randint(1, 10),     # count
            random.randint(1, 10),     # srv_count
            random.uniform(0.0, 0.05), # serror_rate
            random.uniform(0.0, 0.05), # srv_serror_rate
            0                          # attack_type (Normal)
        ]

    elif attack == 1:  # DDoS
        # Extremely high count and srv_count
        return [
            0,                         # duration
            "tcp",                     # protocol_type (often tcp/icmp floods)
            0,                         # src_bytes
            0,                         # dst_bytes
            0,                         # failed_logins
            0,                         # logged_in
            random.randint(200, 500),  # count (VERY HIGH)
            random.randint(200, 500),  # srv_count (VERY HIGH)
            random.uniform(0.8, 1.0),  # serror_rate (VERY HIGH)
            random.uniform(0.8, 1.0),  # srv_serror_rate (VERY HIGH)
            1                          # attack_type (DDoS)
        ]

    elif attack == 2:  # Brute Force
        # High failed_logins, high count, low traffic volume
        return [
            0,                         # duration
            "tcp",                     # protocol_type (usually TCP like SSH/FTP)
            random.randint(50, 150),   # src_bytes (auth attempts are small)
            random.randint(50, 200),   # dst_bytes
            random.randint(10, 50),    # failed_logins (VERY HIGH)
            0,                         # logged_in (usually fail)
            random.randint(20, 100),   # count
            random.randint(5, 20),     # srv_count
            random.uniform(0.0, 0.2),  # serror_rate
            random.uniform(0.0, 0.2),  # srv_serror_rate
            2                          # attack_type (Brute Force)
        ]

    elif attack == 3:  # Probe
        # High count of different services, port scanning
        return [
            random.randint(1, 5),      # duration
            random.choice(["tcp", "icmp"]), # protocol_type
            random.randint(10, 50),    # src_bytes
            random.randint(10, 50),    # dst_bytes
            0,                         # failed_logins
            0,                         # logged_in
            random.randint(50, 150),   # count
            random.randint(1, 5),      # srv_count (low srv_count compared to count -> high diff_srv_rate conceptually)
            random.uniform(0.3, 0.7),  # serror_rate
            random.uniform(0.3, 0.7),  # srv_serror_rate
            3                          # attack_type (Probe)
        ]

    elif attack == 4:  # R2L (Remote to Local)
        # Login attempts from outside, low volume
        return [
            random.randint(5, 20),     # duration
            "tcp",                     # protocol_type
            random.randint(10, 100),   # src_bytes
            random.randint(10, 100),   # dst_bytes
            random.randint(1, 5),      # failed_logins
            1,                         # logged_in (sometimes successful in guessing)
            random.randint(1, 10),     # count
            random.randint(1, 10),     # srv_count
            random.uniform(0.0, 0.1),  # serror_rate
            random.uniform(0.0, 0.1),  # srv_serror_rate
            4                          # attack_type (R2L)
        ]

    elif attack == 5:  # U2R (User to Root)
        # Abnormal patterns after login, specific small bytes sizes
        return [
            random.randint(1, 5),      # duration
            "tcp",                     # protocol_type
            random.randint(30, 80),    # src_bytes (exploit payload)
            random.randint(100, 400),  # dst_bytes
            0,                         # failed_logins
            1,                         # logged_in (Crucial for U2R)
            random.randint(1, 5),      # count (low to evade detection)
            random.randint(1, 5),      # srv_count
            random.uniform(0.0, 0.1),  # serror_rate
            random.uniform(0.0, 0.1),  # srv_serror_rate
            5                          # attack_type (U2R)
        ]

rows = []
TOTAL_ROWS = 10000
current_row = 0

print("Generating stream dataset with structured continuous time-blocks...")

while current_row < TOTAL_ROWS:
    # 35% chance to have a normal block, 65% chance for some attack block
    attack = random.choice([0, 0, 0, 0, 1, 2, 3, 4, 5])
    
    if attack == 0:
        # Normal traffic burst
        block_size = random.randint(50, 300)
    elif attack == 1:
        # DDoS traffic burst (very long)
        block_size = random.randint(100, 500)
    elif attack == 2:
        # Brute Force attack burst
        block_size = random.randint(40, 150)
    elif attack == 3:
        # Probe attack burst
        block_size = random.randint(30, 100)
    else:
        # R2L and U2R are smaller burst attempts
        block_size = random.randint(5, 20)
        
    # Cap block size if near end
    block_size = min(block_size, TOTAL_ROWS - current_row)
    
    for _ in range(block_size):
        row = generate_row(attack)
        rows.append(row)
        current_row += 1

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
    "srv_serror_rate",
    "attack_type"
]

df = pd.DataFrame(rows, columns=columns)
df.to_csv(r"K:\LSTM\data\stream_dataset.csv", index=False)
print(f"Dataset generated successfully with {len(df)} records: stream_dataset.csv")