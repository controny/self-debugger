from self_debugger import SelfDebugger
from dataset_loader import MBPPDatasetLoader

if __name__ == "__main__":
    # Load the MBPP dataset
    dataset_loader = MBPPDatasetLoader()
    problems = dataset_loader.get_all_problems('test')

    # Initialize the SelfDebugger
    self_debugger = SelfDebugger()

    # Iterate through the problems, debug the code, and check if it passes the test cases
    for i, problem in enumerate(problems):
        description = problem['description']
        test_list = problem['test_list']
        generated_code, passed_tests = self_debugger.debug_code(description, test_list)
        
        print(f"Problem {i+1}:")
        print(f"Description: {description}")
        print("Generated Code:", generated_code)
        print("Passed All Tests:", "Yes" if passed_tests else "No")
        print("-" * 50)