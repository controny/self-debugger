import json
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import get_context
import traceback
import argparse
import time
from openai import RateLimitError, BadRequestError
from self_debugger import SelfDebugger
from dataset_loader import ClassEvalDatasetLoader


def process_problem(inp):
    problem, args = inp
    self_debugger = SelfDebugger(args.model_name,
                                max_debugging_steps=args.max_debugging_steps,
                                temperature=args.temperature)

    num_retries = 3
    for _ in range(num_retries):
        try:
            result = self_debugger.debug_code(problem)
            result.update(problem)
            return result
        except RateLimitError as e:
            print("Problem:", problem['task_id'])
            print(f"Rate Limit Error: {e}")
            time.sleep(60)
        except BadRequestError as e:
            if 'context_length_exceeded' in str(e):
                print("Problem:", problem['task_id'])
                print("Context length exceeded, retrying with a smaller max debugging step.")
                self_debugger.max_debugging_steps = int(self_debugger.max_debugging_steps * 0.6)
            else:
                traceback.print_exc()
                break
        except Exception as e:
            traceback.print_exc()
            break
    print("Failed to process problem", problem['task_id'])
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_debugging_steps', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    # Load the dataset
    dataset_loader = ClassEvalDatasetLoader()
    problems = dataset_loader.get_all_problems()
    if args.debug:
        problems = problems[:5]
    print(f"Number of original problems: {len(problems)}")

    # Iterate through the problems, debug the code, and check if it passes the test cases
    num_success = 0
    num_total = len(problems)

    save_path = 'outputs/classeval/results.jsonl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed_task_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as jsonfile:
            for line in jsonfile:
                result = json.loads(line)
                processed_task_ids.add(result['task_id'])
                # also update the success rate
                num_success += int(result['success'])
    
    # Filter out the processed problems
    problems = [problem for problem in problems if problem['task_id'] not in processed_task_ids]
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
