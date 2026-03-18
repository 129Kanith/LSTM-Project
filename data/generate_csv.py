import pandas as pd
import random

rows = []

for i in range(10000):

    attack = random.choice([
        "normal",
        "ddos",
        "bruteforce",
        "probe",
        "r2l",
        "u2r"
    ])

    if attack == "normal":

        row = [
            random.randint(3,10),
            random.choice(["tcp","udp"]),
            random.choice(["http","dns","smtp"]),
            random.randint(200,600),
            random.randint(1500,5000),
            0,
            random.randint(1,5),
            random.randint(1,5),
            1.0,
            0.0,
            0
        ]

    elif attack == "ddos":

        row = [
            0,
            "tcp",
            "http",
            0,
            0,
            0,
            random.randint(150,400),
            random.randint(140,380),
            random.uniform(0.02,0.15),
            random.uniform(0.85,0.98),
            1
        ]

    elif attack == "bruteforce":

        row = [
            0,
            "tcp",
            "ssh",
            random.randint(120,200),
            random.randint(700,1000),
            random.randint(5,12),
            random.randint(20,40),
            random.randint(15,30),
            random.uniform(0.5,0.8),
            random.uniform(0.2,0.5),
            2
        ]

    elif attack == "probe":

        row = [
            random.randint(2,6),
            "tcp",
            random.choice(["ftp","smtp"]),
            random.randint(80,160),
            random.randint(200,500),
            random.randint(1,3),
            random.randint(8,15),
            random.randint(6,12),
            random.uniform(0.4,0.7),
            random.uniform(0.3,0.6),
            3
        ]

    elif attack == "r2l":

        row = [
            0,
            "tcp",
            "http",
            0,
            0,
            random.randint(8,15),
            random.randint(30,60),
            random.randint(25,50),
            random.uniform(0.3,0.6),
            random.uniform(0.4,0.7),
            4
        ]

    elif attack == "u2r":

        row = [
            random.randint(2,5),
            "tcp",
            "ssh",
            random.randint(40,120),
            random.randint(200,400),
            random.randint(1,4),
            random.randint(10,20),
            random.randint(8,15),
            random.uniform(0.45,0.65),
            random.uniform(0.35,0.55),
            5
        ]

    rows.append(row)


columns = [
    "duration",
    "protocol_type",
    "service",
    "src_bytes",
    "dst_bytes",
    "failed_logins",
    "count",
    "srv_count",
    "same_srv_rate",
    "diff_srv_rate",
    "attack_type"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv(r"K:\LSTM\data\10000.csv", index=False)

print("Dataset generated successfully: 10000.csv")