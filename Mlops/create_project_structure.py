import os

# Folder names to create
folders = [
    "data/raw", "data/processed",
    "src/ingestion", "src/preprocessing", "src/features",
    "src/models", "src/serving", "src/monitoring", "src/utils", "src/webapp",
    "artifacts", "notebooks", "logs", "docker", "ci"
]

# Key files to create
files = [
    "requirements.txt", "README.md", ".env",
    "docker/Dockerfile",
    "ci/github-actions.yml"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    # Ensure parent directory exists before writing the file
    parent = os.path.dirname(file)
    if parent != "" and not os.path.exists(parent):
        os.makedirs(parent)
    with open(file, "w") as f:
        f.write("")

print("✅ All folders and files created in your VS Code workspace!")
