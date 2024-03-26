from datasets import load_dataset

class MBPPDatasetLoader:
    def __init__(self):
        """
        Initialize the dataset loader and load the MBPP dataset from HuggingFace.
        """
        self.dataset = load_dataset("mbpp")

    def get_problem(self, split, index):
        """
        Get a single problem by its index from the specified split.

        :param split: Split of the dataset to retrieve the problem from ('train', 'test', or 'validation').
        :param index: Index of the problem to retrieve.
        :return: A dict representing the problem, including its description, solution, and test_list.
        """
        problem = self.dataset[split][index]
        return {
            "description": problem['text'],
            "solution": problem['code'],
            "test_list": problem['test_list']  # Assumes 'test_list' is available
        }

    def get_all_problems(self, split):
        """
        Get all problems from the specified split.

        :param split: Split of the dataset to retrieve problems from ('train', 'test', or 'validation').
        :return: A list of dicts, each representing a problem including description, solution, and test_list.
        """
        problems = self.dataset[split]
        return [{
            "description": problem['text'],
            "solution": problem['code'],
            "test_list": problem['test_list']  # Assumes 'test_list' is available
        } for problem in problems]


if __name__ == "__main__":
    # Example usage:
    dataset_loader = MBPPDatasetLoader()

    # To iterate over all test problems and their test cases:
    for problem in dataset_loader.get_all_problems('test'):
        print(problem['description'])
        print(problem['solution'])
        print(problem['test_list'])
        break
