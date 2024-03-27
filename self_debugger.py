from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class SelfDebugger:
    def __init__(self, model_name='gpt-3.5-turbo', max_debugging_steps=10):
        """
        Initialize the self-debugger.

        :param model_name: The name of the language model to use.
        :param max_debugging_steps: Maximum number of debugging iterations.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)  # use greedy decoding
        template = "Complete the following task in Python:\n{description}\nThis is of the assertions for your function:\n`{assertion}`\nReturn the complete and correct function code with no additional text or statement."
        prompt = PromptTemplate(template=template, input_variables=["description", "assertion"])
        # self.generator = CodeChain.from_prompt(prompt, self.llm)
        self.generator = LLMChain(llm=self.llm, prompt=prompt)
        self.max_debugging_steps = max_debugging_steps

    def generate_code(self, problem_description, assertion):
        """
        Generate initial code based on the problem description.

        :param problem_description: The problem description.
        :param assertion: Assertion for the function.
        :return: Generated code.
        """
        res = self.generator.invoke(
            dict(description=problem_description, assertion=assertion)
            )
        return res['text']

    def execute_code(self, code, test_list):
        """
        Execute the given code and return execution results.

        :param code: The code to execute.
        :param test_list: List of test cases.
        :return: Execution results.
        """
        # Implementation using the executor
        pass

    def generate_feedback(self, execution_results):
        """
        Generate feedback based on the execution results.

        :param execution_results: The results of code execution.
        :return: Feedback message.
        """
        # Implementation using LLM to generate feedback
        pass

    def debug_code(self, problem_description, test_list):
        """
        Main debugging loop.

        :param problem_description: The problem description.
        :param test_list: List of test cases.
        :return: Debugged code.
        """
        code = self.generate_code(problem_description, test_list[0])
        print("Initial Code:")
        print(code)
        for _ in range(self.max_debugging_steps):
            execution_results = self.execute_code(code, test_list)
            if self.is_solution_correct(execution_results):
                return code  # Return the correct code
            feedback = self.generate_feedback(execution_results)
            code = self.refine_code(code, feedback)
        return code  # Return the best attempt

    def is_solution_correct(self, execution_results):
        """
        Check if the solution meets the criteria based on execution results.

        :param execution_results: The results of code execution.
        :return: Boolean indicating if the solution is correct.
        """
        # Implementation to determine correctness
        pass

    def refine_code(self, code, feedback):
        """
        Refine the code based on feedback.

        :param code: The current version of the code.
        :param feedback: Feedback message.
        :return: Refined code.
        """
        # Implementation to refine code using LLM
        pass


if __name__ == "__main__":
    # Example usage
    self_debugger = SelfDebugger(max_debugging_steps=0)
    problem_description = "Write a python function to remove first and last occurrence of a given character from the string."
    test_list = ['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"', 'assert remove_Occ("PHP","P") == "H"']
    debugged_code = self_debugger.debug_code(problem_description, test_list)
