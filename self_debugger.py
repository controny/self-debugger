import subprocess
import openai
import re
from string import Template
import logging
from func_timeout import func_set_timeout
import importlib
import unittest
import io
import os
import sys
import inspect
from collections import defaultdict
import traceback


SYS_PROMPT = "You are an expert programming assistant."
EXPLAIN_PROMPT = "Explain the Python code line by line."
CODE_GEN_PROMPT_TEMPLATE = Template("Complete the class in the following code:\n${description}\nReturn the pure function code with no additional text or markup.")
UT_FEEDBACK_PROMPT_TEMPLATE = Template("The code above fails the given unit tests:\n${exec_res}\nPlease fix the Python code.")

class SelfDebugger:

    def __init__(self, model_name='gpt-3.5-turbo', max_debugging_steps=10, temperature=0.0):
        """
        Initialize the self-debugger.

        :param model_name: The name of the language model to use.
        :param max_debugging_steps: Maximum number of debugging iterations.
        :param temperature: The temperature parameter for the language model.
        """
        self.model_name = model_name
        self.llm_client = openai.OpenAI()
        self.history = []
        self.max_debugging_steps = max_debugging_steps
        self.temperature = temperature
        self.cur_task = None
        self.log_dir = 'outputs/classeval'
        os.makedirs(self.log_dir, exist_ok=True)
        # Get the absolute path of the directory of the module
        module_dir = os.path.abspath(self.log_dir)
        # Add the directory of the module to the system path to enable importing the generated module
        sys.path.insert(0, module_dir)
    
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
            messages=self.history,
            temperature=self.temperature,
        )
        response_text = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": response_text})
        return response_text
    
    def add_static_statement(self, code):
        filtered_code_list = []
        for line in code.split('\n'):
            if '@staticmethod' in line:
                continue
            filtered_code_list.append(line)
        code = '\n'.join(filtered_code_list)
        final_code_list = []
        for line in code.split('\n'):
            if line.strip().startswith('def ') and 'self' not in line and 'cls' not in line and self.get_leading_spaces(line) == 4:
                final_code_list.append('    @staticmethod')
            final_code_list.append(line)
        return '\n'.join(final_code_list)
    
    def extract_code(self, text):
        """
        Extract code from the given text.

        :param text: The text to extract code from.
        :return: Extracted code.
        """
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
            code = self.add_static_statement(code)
            code = '\n'.join(self.cur_task['import_statement']) + '\n' + code
            return code
        else:
            return ''

    def generate_code(self):
        """
        Generate initial code based on the problem description.

        :return: Generated code.
        """
        # reset history
        self.history = [
            {"role": "system", "content": SYS_PROMPT},
        ]
        prompt = CODE_GEN_PROMPT_TEMPLATE.substitute(description=self.cur_task['skeleton'])
        logging.debug("Code Generation Prompt:")
        logging.debug(prompt)
        response_text = self.get_response_text(prompt)
        return self.extract_code(response_text)
    
    def gen_py_file(self, test_code_name, code_snippet, test_code):
        test_code_py = code_snippet + '\n\n' + test_code
        with open(os.path.join(self.log_dir, test_code_name + '.py'), 'w', encoding='utf-8') as f:
            f.write(test_code_py)

    def run_unit_test(self, module_name, test_class, log_path):
        module = importlib.import_module(module_name)
        test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_class))
        with open(log_path, 'a', encoding='utf-8') as f:
            test_res = unittest.TextTestRunner(stream=f).run(test_suite)
        exec_res = dict()
        for test_case_obj, trace in test_res.failures + test_res.failures:
            test_method_name = test_case_obj.id().split('.')[-1]
            code = inspect.getsource(getattr(type(test_case_obj), test_method_name))
            error = trace.strip().split('\n')[-1]
            exec_res[code] = error
        num_tests = test_res.testsRun

        return exec_res, num_tests

    def test(self, module_name, test_classes):
        res = dict()  # map from test code to error message
        log_path = os.path.join(self.log_dir, module_name + '.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        self.cur_task['num_tests'] = 0
        num_passes = 0
        for test_class in test_classes:
            try:
                fail_res, num_tests = self.run_unit_test(module_name, test_class, log_path)
                res.update(fail_res)
                self.cur_task['num_tests'] += num_tests
                num_passes += num_tests - len(fail_res)
            except:
                traceback.print_exc()
        self.cur_task['num_passes_list'].append(num_passes)

        return res

    def execute_code(self, code, iter):
        """
        Execute the given code and return execution results.

        :param code: The code to execute.
        :param iter: The current iteration.
        :return: Execution results.
        """
        generated_module_name = self.cur_task['task_id'] + '-' + str(iter)
        self.gen_py_file(generated_module_name, code, self.cur_task['test'])
        outputs = self.test(generated_module_name, self.cur_task['test_classes'])
        return outputs

    def refine_code(self, execution_results):
        """
        Refine code based on the execution results.

        :param execution_results: The results of code execution.
        :return: Refinement result.
        """
        explanation = self.get_response_text(EXPLAIN_PROMPT)
        exec_res = '\n'.join([f"```\n{code}\n```\n{error}" for code, error in execution_results.items()])
        feedback_prompt = UT_FEEDBACK_PROMPT_TEMPLATE.substitute(exec_res=exec_res)
        logging.debug("Feedback Prompt:")
        logging.debug(feedback_prompt)
        refined_code = self.extract_code(self.get_response_text(feedback_prompt))
        return refined_code, explanation

    def debug_code(self, task):
        """
        Main debugging loop.

        :param problem_description: The problem description.
        :param test_list: List of test cases.
        :return: Result of debugging.
        """
        success = False
        refined = False  # whether the code has been refined
        self.cur_task = task
        # keep track of the number of passes for each iteration
        self.cur_task['num_passes_list'] = []
        code = self.generate_code()
        initial_code = code
        explanation = ''
        logging.debug("Initial Code:")
        logging.debug(code)
        for i in range(self.max_debugging_steps):
            execution_results = self.execute_code(code, i)
            logging.debug("Execution Results:")
            logging.debug(execution_results)
            if self.is_solution_correct(execution_results):
                success = True
                break
            # for the last iteration, no need to refine the code
            if i == self.max_debugging_steps - 1:
                break
            code, explanation = self.refine_code(execution_results)
            refined = True
            logging.debug("Refined Code:")
            logging.debug(code)
        res = {
            'initial_code': initial_code,
            'refined_code': code,
            'execution_results': execution_results,
            'explanation': explanation,
            'success': success,
            'refined': refined,
            'num_passes_list': self.cur_task['num_passes_list'],
            'num_tests': self.cur_task['num_tests'],
            'iterations': i + 1,
        }
        logging.debug("Result:")
        logging.debug(res)
        return res

    def is_solution_correct(self, execution_results):
        """
        Check if the solution meets the criteria based on execution results.

        :param execution_results: The results of code execution.
        :return: Boolean indicating if the solution is correct.
        """
        return len(execution_results) == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Example usage
    self_debugger = SelfDebugger(max_debugging_steps=3)
    task = {
        "task_id": "ClassEval_0",
        "skeleton": "import logging\nimport datetime\n\nclass AccessGatewayFilter:\n    \"\"\"\n    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.\n    \"\"\"\n\n    def __init__(self):\n        pass\n\n    def filter(self, request):\n        \"\"\"\n        Filter the incoming request based on certain rules and conditions.\n        :param request: dict, the incoming request details\n        :return: bool, True if the request is allowed, False otherwise\n        >>> filter = AccessGatewayFilter()\n        >>> filter.filter({'path': '/login', 'method': 'POST'})\n        True\n\n        \"\"\"\n\n\n    def is_start_with(self, request_uri):\n        \"\"\"\n        Check if the request URI starts with certain prefixes.\n        :param request_uri: str, the URI of the request\n        :return: bool, True if the URI starts with certain prefixes, False otherwise\n        >>> filter = AccessGatewayFilter()\n        >>> filter.is_start_with('/api/data')\n        True\n\n        \"\"\"\n\n\n    def get_jwt_user(self, request):\n        \"\"\"\n        Get the user information from the JWT token in the request.\n        :param request: dict, the incoming request details\n        :return: dict or None, the user information if the token is valid, None otherwise\n        >>> filter = AccessGatewayFilter()\n        >>> filter.get_jwt_user({'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1'+str(datetime.date.today())}}})\n        {'user': {'name': 'user1'}\n\n        \"\"\"\n\n    def set_current_user_info_and_log(self, user):\n        \"\"\"\n        Set the current user information and log the access.\n        :param user: dict, the user information\n        :return: None\n        >>> filter = AccessGatewayFilter()\n        >>> user = {'name': 'user1', 'address': '127.0.0.1'}\n        >>> filter.set_current_user_info_and_log(user)\n\n        \"\"\"",
        "test": "import unittest\n\nclass AccessGatewayFilterTestFilter(unittest.TestCase):\n    def test_filter_1(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/api/data', 'method': 'GET'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_2(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/api/data', 'method': 'POST'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_3(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/login/data', 'method': 'GET'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_4(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/login/data', 'method': 'POST'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_5(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/abc', 'method': 'POST',\n                   'headers': {\n                       'Authorization': {'user': {'name': 'user1', 'level': 5, 'address': 'address1'},\n                                         'jwt': 'user1' + str(datetime.date.today())}}}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_6(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/abc', 'method': 'POST',\n                   'headers': {\n                       'Authorization': {'user': {'name': 'user1', 'level': 3, 'address': 'address1'},\n                                         'jwt': 'user1' + str(datetime.date.today() - datetime.timedelta(days=365))}}}\n        res = agf.filter(request)\n        self.assertFalse(res)\n\n    def test_filter_7(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/abc', 'method': 'POST',\n                   'headers': {\n                       'Authorization': {'user': {'name': 'user1', 'level': 1, 'address': 'address1'},\n                                         'jwt': 'user1' + str(datetime.date.today())}}}\n        res = agf.filter(request)\n        self.assertIsNone(res)\n\n    def test_filter_8(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/abc', 'method': 'POST',\n                   'headers': {\n                       'Authorization': {'user': {'name': 'user1', 'level': 3, 'address': 'address1'},\n                                         'jwt': 'user2' + str(datetime.date.today() - datetime.timedelta(days=365))}}}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n\nclass AccessGatewayFilterTestIsStartWith(unittest.TestCase):\n    def test_is_start_with_1(self):\n        agf = AccessGatewayFilter()\n        request_uri = '/api/data'\n        res = agf.is_start_with(request_uri)\n        self.assertTrue(res)\n\n    def test_is_start_with_2(self):\n        agf = AccessGatewayFilter()\n        request_uri = '/admin/settings'\n        res = agf.is_start_with(request_uri)\n        self.assertFalse(res)\n\n    def test_is_start_with_3(self):\n        agf = AccessGatewayFilter()\n        request_uri = '/login/data'\n        res = agf.is_start_with(request_uri)\n        self.assertTrue(res)\n\n    def test_is_start_with_4(self):\n        agf = AccessGatewayFilter()\n        request_uri = '/abc/data'\n        res = agf.is_start_with(request_uri)\n        self.assertFalse(res)\n\n    def test_is_start_with_5(self):\n        agf = AccessGatewayFilter()\n        request_uri = '/def/data'\n        res = agf.is_start_with(request_uri)\n        self.assertFalse(res)\n\n\nclass AccessGatewayFilterTestGetJwtUser(unittest.TestCase):\n    def test_get_jwt_user_1(self):\n        agf = AccessGatewayFilter()\n        request = {\n            'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1' + str(datetime.date.today())}}}\n        res = agf.get_jwt_user(request)\n        self.assertIsNotNone(res)\n\n    def test_get_jwt_user_2(self):\n        agf = AccessGatewayFilter()\n        request = {\n            'headers': {'Authorization': {'user': {'name': 'user2'}, 'jwt': 'user2' + str(datetime.date.today())}}}\n        res = agf.get_jwt_user(request)\n        self.assertIsNotNone(res)\n\n    def test_get_jwt_user_3(self):\n        agf = AccessGatewayFilter()\n        request = {\n            'headers': {'Authorization': {'user': {'name': 'user3'}, 'jwt': 'user3' + str(datetime.date.today())}}}\n        res = agf.get_jwt_user(request)\n        self.assertIsNotNone(res)\n\n    def test_get_jwt_user_4(self):\n        agf = AccessGatewayFilter()\n        request = {\n            'headers': {'Authorization': {'user': {'name': 'user4'}, 'jwt': 'user4' + str(datetime.date.today())}}}\n        res = agf.get_jwt_user(request)\n        self.assertIsNotNone(res)\n\n    def test_get_jwt_user_5(self):\n        agf = AccessGatewayFilter()\n        request = {'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1' + str(\n            datetime.date.today() - datetime.timedelta(days=5))}}}\n        res = agf.get_jwt_user(request)\n        self.assertIsNone(res)\n\n\nclass AccessGatewayFilterTest(unittest.TestCase):\n    def test_AccessGatewayFilter(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/api/data', 'method': 'GET'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n        request_uri = '/api/data'\n        res = agf.is_start_with(request_uri)\n        self.assertTrue(res)\n\n        request = {\n            'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1' + str(datetime.date.today())}}}\n        res = agf.get_jwt_user(request)\n        self.assertIsNotNone(res)",
        "solution_code": "import logging\nimport datetime\n\n\nclass AccessGatewayFilter:\n\n    def __init__(self):\n        pass\n\n    def filter(self, request):\n        request_uri = request['path']\n        method = request['method']\n\n        if self.is_start_with(request_uri):\n            return True\n\n        try:\n            token = self.get_jwt_user(request)\n            user = token['user']\n            if user['level'] > 2:\n                self.set_current_user_info_and_log(user)\n                return True\n        except:\n            return False\n\n    def is_start_with(self, request_uri):\n        start_with = [\"/api\", '/login']\n        for s in start_with:\n            if request_uri.startswith(s):\n                return True\n        return False\n\n    def get_jwt_user(self, request):\n        token = request['headers']['Authorization']\n        user = token['user']\n        if token['jwt'].startswith(user['name']):\n            jwt_str_date = token['jwt'].split(user['name'])[1]\n            jwt_date = datetime.datetime.strptime(jwt_str_date, \"%Y-%m-%d\")\n            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):\n                return None\n        return token\n\n    def set_current_user_info_and_log(self, user):\n        host = user['address']\n        logging.log(msg=user['name'] + host + str(datetime.datetime.now()), level=1)",
        "import_statement": [
            "import logging",
            "import datetime"
        ],
        "class_description": "    \"\"\"\n    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.\n    \"\"\"\n",
        "class_name": "AccessGatewayFilter",
        "test_classes": [
            "AccessGatewayFilterTestFilter",
            "AccessGatewayFilterTestIsStartWith",
            "AccessGatewayFilterTestGetJwtUser",
            "AccessGatewayFilterTest"
        ],
        "class_constructor": "class AccessGatewayFilter: \n    def __init__(self):\n        pass\n\n",
        "fields": [],
    }
    debugged_code = self_debugger.debug_code(task)
