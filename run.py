import json
import os
from tqdm import tqdm
import argparse
from self_debugger import SelfDebugger
from dataset_loader import MBPPDatasetLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_debugging_steps', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    # Load the MBPP dataset
    dataset_loader = MBPPDatasetLoader()
    problems = dataset_loader.get_all_problems('test')
    if args.debug:
        problems = problems[:5]
    print(f"Number of problems: {len(problems)}")

    # Initialize the SelfDebugger
    self_debugger = SelfDebugger(args.model_name,
                                max_debugging_steps=args.max_debugging_steps,
                                temperature=args.temperature)

    # TODO use multiprocessing to speed up the process
    # Iterate through the problems, debug the code, and check if it passes the test cases
    num_success = 0

    save_path = 'outputs/results.jsonl'
    processed_indices = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as jsonfile:
            for line in jsonfile:
                result = json.loads(line)
                processed_indices.add(result['index'])
                # also update the success rate
                num_success += int(result['success'])

    with open(save_path, 'a') as jsonfile:
        for i, problem in tqdm(enumerate(problems)):
            if i in processed_indices:
                continue

            description = problem['description']
            test_list = problem['test_list']
            result = self_debugger.debug_code(description, test_list)
            num_success += int(result['success'])

            result.update({
                'index': i,
                'description': description,
                'test_list': test_list,
            })
            jsonfile.write(json.dumps(result) + '\n')
            jsonfile.flush()

    print(f"Success Rate: {num_success}/{len(problems)}")