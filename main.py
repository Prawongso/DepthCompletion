import subprocess
import sys

# Define the path to the conda environment and the script
conda_env_name = "completion"  # Replace with your Conda environment name
script1 = r"ui/Pluralistic-Inpainting-master/test.py"  # Path to the first script

# Step 1: Run test.py in the different Conda environment
try:
    subprocess.run(f"conda activate {conda_env_name} && python {script1}", 
                   shell=True, check=True)
    print(f"Successfully ran {script1} in {conda_env_name}")
except subprocess.CalledProcessError as e:
    print(f"Failed to run {script1}: {e}")

# Step 2: Run the second script as a module using "python -m"
module_name = "ui.mymain"  # Replace with your module path
try:
    subprocess.run([sys.executable, "-m", module_name], check=True)
    print(f"Successfully ran {module_name} as a module in the current environment")
except subprocess.CalledProcessError as e:
    print(f"Failed to run {module_name}: {e}")
