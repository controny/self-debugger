import subprocess
import openai
import re
from string import Template
import logging

SYS_PROMPT = "You are an expert programming assistant."
EXPLAIN_PROMPT = "Explain the Python code line by line."
CODE_GEN_PROMPT_TEMPLATE = Template("Complete the following task in Python:\n${description}\nThis is of the assertions for your function:\n`${assertion}`\nReturn the pure function code with no additional text or markup.")
UT_FEEDBACK_PROMPT_TEMPLATE = Template("The code above fails the given unit test:\n${exec_res}\nPlease fix the Python code.")

class SelfDebugger:

    def __init__(self, model_name='gpt-3.5-turbo', max_debugging_steps=10):
        """
        Initialize the self-debugger.

        :param model_name: The name of the language model to use.
        :param max_debugging_steps: Maximum number of debugging iterations.
        """
        self.model_name = model_name
        self.llm_client = openai.OpenAI()
        self.history = []
        self.max_debugging_steps = max_debugging_steps
    
    def get_response_text(self, message):
        """
        Send message to the language model and parse the response to get text.
        Update the history at the same time.

        :param message: The message string.
        :return: Response text.
        """
        self.history.append({"role": "user", "content": message})
        resp = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=self.history
        )
        response_text = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": response_text})
        return response_text
    
    def extract_code(self, text):
        """
        Extract code from the given text.

        :param text: The text to extract code from.
        :return: Extracted code.
        """
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return ''

    def generate_code(self, problem_description, assertion):
        """
        Generate initial code based on the problem description.

        :param problem_description: The problem description.
        :param assertion: Assertion for the function.
        :return: Generated code.
        """
        # reset history
        self.history = [
            {"role": "system", "content": SYS_PROMPT},
        ]
        prompt = CODE_GEN_PROMPT_TEMPLATE.substitute(description=problem_description, assertion=assertion)
        response_text = self.get_response_text(prompt)
        return self.extract_code(response_text)

    def execute_code(self, code, test_list):
        """
        Execute the given code and return execution results.

        :param code: The code to execute.
        :param test_list: List of test cases.
        :return: Execution results.
        """
        outputs = dict()
        for test in test_list:
            code_str = code + '\n' + test
            result = subprocess.run(
                ['python3', '-c', code_str],
                text=True,
                capture_output=True  # Captures stdout and stderr from the subprocess
            )

            # Checking the result
            if result.returncode == 0:
                # use empty string to indicate success
                outputs[test] = ''
            else:
                outputs[test] = result.stderr

        return outputs

    def refine_code(self, execution_results):
        """
        Generate feedback based on the execution results.

        :param execution_results: The results of code execution.
        :return: Feedback message.
        """
        explanation = self.get_response_text(EXPLAIN_PROMPT)
        exec_res = '\n'.join([f"`{test}`: `{result}`" for test, result in execution_results.items()])
        feedback_prompt = UT_FEEDBACK_PROMPT_TEMPLATE.substitute(exec_res=exec_res)
        refined_code = self.extract_code(self.get_response_text(feedback_prompt))
        return refined_code

    def debug_code(self, problem_description, test_list):
        """
        Main debugging loop.

        :param problem_description: The problem description.
        :param test_list: List of test cases.
        :return: Debugged code.
        """
        code = self.generate_code(problem_description, test_list[0])
        logging.debug("Initial Code:")
        logging.debug(code)
        for _ in range(self.max_debugging_steps):
            execution_results = self.execute_code(code, test_list)
            logging.debug("Execution Results:")
            logging.debug(execution_results)
            if self.is_solution_correct(execution_results):
                break
            code = self.refine_code(execution_results)
            logging.debug("Refined Code:")
            logging.debug(code)
        logging.debug("History:")
        logging.debug('\n'.join([str(item) for item in self.history]))
        return code  # Return the best attempt

    def is_solution_correct(self, execution_results):
        """
        Check if the solution meets the criteria based on execution results.

        :param execution_results: The results of code execution.
        :return: Boolean indicating if the solution is correct.
        """
        return all([result == '' for result in execution_results.values()])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Example usage
    self_debugger = SelfDebugger(max_debugging_steps=3)
    problem_description = "Write a python function to remove first and last occurrence of a given character from the string."
    test_list = ['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"', 'assert remove_Occ("PHP","P") == "H"']
    debugged_code = self_debugger.debug_code(problem_description, test_list)
