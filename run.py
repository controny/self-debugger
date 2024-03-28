import json
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import get_context
import traceback
import argparse
import time
from openai import RateLimitError
from self_debugger import SelfDebugger
from dataset_loader import MBPPDatasetLoader


def process_problem(inp):
    problem, args = inp
    description = problem['description']
    test_list = problem['test_list']
    
    self_debugger = SelfDebugger(args.model_name,
                                max_debugging_steps=args.max_debugging_steps,
                                temperature=args.temperature)

    num_retries = 3
    for _ in range(num_retries):
        try:
            result = self_debugger.debug_code(description, test_list)
            result.update({
                'description': description,
                'test_list': test_list,
            })
            return result
        except RateLimitError as e:
            print(f"Rate Limit Error: {e}")
            time.sleep(60)
        except Exception as e:
            traceback.print_exc()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_debugging_steps', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    # Load the MBPP dataset
    dataset_loader = MBPPDatasetLoader()
    problems = dataset_loader.get_all_problems('test')
    if args.debug:
        problems = problems[:5]
    print(f"Number of original problems: {len(problems)}")

    # Iterate through the problems, debug the code, and check if it passes the test cases
    num_success = 0
    num_total = len(problems)

    save_path = 'outputs/results.jsonl'
    processed_desc = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as jsonfile:
            for line in jsonfile:
                result = json.loads(line)
                processed_desc.add(result['description'])
                # also update the success rate
                num_success += int(result['success'])
    
    # Filter out the processed problems
    problems = [problem for problem in problems if problem['description'] not in processed_desc]
    print(f"Number of unprocessed problems: {len(problems)}")

    num_workers = args.num_workers
    print(f"Number of workers: {num_workers}")
    with multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=len(problems)) as pbar:
            for result in pool.imap_unordered(process_problem, [(problem, args) for problem in problems]):
                pbar.update()
                if result is not None:
                    with open(save_path, 'a') as jsonfile:
                        num_success += int(result['success'])
                        jsonfile.write(json.dumps(result) + '\n')
                        jsonfile.flush()
        pool.close()
        pool.join()

    print(f"Success Rate: {num_success}/{num_total} = {num_success / num_total:.4f}")
