import os
import subprocess

def install_requirements(Rambutan-Oracle-Sweet-Cravings-Decoded):
    """
    Install dependencies from requirements.txt within a specific folder.
    """
    requirements_path = os.path.join(folder_name, "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call(["pip", "install", "-r", requirements_path])
            print(f"Dependencies installed successfully from {requirements_path}.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies from {requirements_path}: {e}")
    else:
        print(f"No requirements.txt found in {folder_name}.")