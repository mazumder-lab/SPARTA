import GPUtil
import subprocess
import time

def get_available_gpu():
    """Return the first available GPU, None if all are in use."""
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        if gpu.load < 0.1:  # Adjust the threshold to your needs
            return gpu.id
    return None

def run_job(command, gpu_id):
    """Run the job on the specified GPU."""
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    subprocess.Popen(command, shell=True, env=env)

def run_all_jobs(jobs):
    # Assign jobs to GPUs
    while jobs:
        gpu_id = get_available_gpu()
        if gpu_id is not None:
            job = jobs.pop(0)
            run_job(job, gpu_id)
        else:
            time.sleep(10)  # Wait for 10 seconds before checking GPU availability again
