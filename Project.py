import os

# Define the project structure
project_structure = {
    "cyberattack_intelligence_system": [
        "app.py",
        "config.py",
        "train.py",
        "evaluate.py",
        "inference.py",
        "predict_single.py",
        "predict_csv.py",
        {"data": [
            "offline_dataset.csv",
            "stream_dataset.csv"
        ]},
        {"models": [
            "lstm_v1.h5",
            "scaler_v1.pkl"
        ]},
        {"core": [
            "preprocessing.py",
            "sequence_builder.py",
            "model_builder.py",
            "metrics.py"
        ]},
        {"logs": [
            "training.log"
        ]}
    ]
}

def create_structure(base_path, structure):
    for item in structure:
        if isinstance(item, str):
            # Handle files and folders
            if item.endswith("/"):
                os.makedirs(os.path.join(base_path, item), exist_ok=True)
            else:
                file_path = os.path.join(base_path, item)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w") as f:
                    f.write("")  # create empty file
        elif isinstance(item, dict):
            for folder, contents in item.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, contents)

# Run the script
base_dir = os.getcwd()  # current working directory
create_structure(base_dir, project_structure["cyberattack_intelligence_system"])

print("Project structure created successfully!")