import csv
from tqdm import tqdm
from self_debugger import SelfDebugger
from dataset_loader import MBPPDatasetLoader

if __name__ == "__main__":
    # Load the MBPP dataset
    dataset_loader = MBPPDatasetLoader()
    problems = dataset_loader.get_all_problems('test')

    # Initialize the SelfDebugger
    self_debugger = SelfDebugger()

    # Iterate through the problems, debug the code, and check if it passes the test cases
    num_success = 0
    with open('outputs/results.csv', 'w', newline='') as csvfile:
        fieldnames = ['description', 'test_list', 'generated_code', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, problem in tqdm(enumerate(problems)):
            description = problem['description']
            test_list = problem['test_list']
            generated_code, success = self_debugger.debug_code(description, test_list)
            num_success += int(success)

            writer.writerow({'description': description, 'test_list': str(test_list), 'generated_code': generated_code, 'success': success})
            csvfile.flush()

    print(f"Success Rate: {num_success}/{len(problems)}")