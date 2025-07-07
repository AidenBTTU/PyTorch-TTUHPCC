import argparse

def get_params(param_file, task_id):
    with open(param_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    if task_id < 0 or task_id >= len(lines):
        raise IndexError(f"Task ID {task_id} out of range.")
    return lines[task_id].split()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--param-file', type=str, default='param_list.txt')
    args = parser.parse_args()

    training, test, results, weights = get_params(args.param_file, args.task_id)

    print(f"Running: {training} {test} {results} {weights}")

    # You can call the real script via subprocess, or import if modular
    import subprocess
    subprocess.run(["python", "ROCAUC.py", training, test, results, weights])

if __name__ == "__main__":
    main()

