import os
import subprocess

# Define folder structure
folders = [
    "notebooks",
    "scripts",
    "data/raw",
    "data/processed",
    "models",
    "reports",
]

files = {
    "README.md": "# Project Overview\n\nThis is the project for building a credit scoring model.",
    "requirements.txt": "pandas\nnumpy\nmatplotlib\nscikit-learn",
    "scripts/data_cleaning.py": "# Script for data cleaning\n\n# Add your data cleaning logic here.",
    "scripts/feature_engineering.py": "# Script for feature engineering\n\n# Add your feature engineering logic here.",
    ".gitignore": "data/raw/*\ndata/processed/*\nmodels/*",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file, content in files.items():
    with open(file, "w") as f:
        f.write(content)

# Initialize Git repository
subprocess.run(["git", "init"])

# Add all files to Git
subprocess.run(["git", "add", "."])

# Commit the changes
subprocess.run(["git", "commit", "-m", "Initial commit"])

# Prompt for GitHub repository details
repo_url = input("Enter the GitHub repository URL (leave blank to skip): ")

# Push to GitHub (if URL is provided)
if repo_url:
    subprocess.run(["git", "remote", "add", "origin", repo_url])
    subprocess.run(["git", "branch", "-M", "main"])
    subprocess.run(["git", "push", "-u", "origin", "main"])
