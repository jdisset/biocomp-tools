import argparse
import subprocess
from pathlib import Path
import logging
from multiprocessing import Pool

def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)

def run_command(filepath, log_dir):
    base_name = filepath.stem
    stderr_log = log_dir / f"{base_name}.stderr"
    stdout_log = log_dir / f"{base_name}.stdout"

    with stderr_log.open('w') as err, stdout_log.open('w') as out:
        command = f"python ~/Code/Weiss/biocomp-tools/biocomptools/plot_data.py --job_file {filepath}"
        logging.info(f"Submitting job for {filepath}")
        process = subprocess.run(command, shell=True, stdout=out, stderr=err)
        return process.returncode, filepath

def main(joblist_file, num_processes, log_dir):
    setup_logging(log_dir)

    with joblist_file.open('r') as file:
        filepaths = [Path(line.strip()) for line in file if line.strip()]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(run_command, [(filepath, log_dir) for filepath in filepaths])
    
    for returncode, filepath in results:
        if returncode != 0:
            logging.error(f"Job failed for {filepath} with return code {returncode}")
        else:
            logging.info(f"Job completed successfully for {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run plot jobs from a list in parallel.')
    parser.add_argument('joblist_file', type=Path, help='Path to the joblist file.')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to run in parallel.')
    parser.add_argument('--log_dir', type=Path, default=Path('./log'), help='Directory to store log files.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    main(args.joblist_file, args.num_processes, args.log_dir)
