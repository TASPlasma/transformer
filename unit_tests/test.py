class OutputTest:
    """
    Basic class for testing output of a function
    matches expected output
    """

    def __init__(self, input, exp_output, func):
        self.input = input
        self.exp_output = exp_output
        self.func = func

    def __call__(self):
        print(f'Expected Output: \n{self.exp_output}, computed output: \n{self.func(self.input)}')
        assert self.exp_output == self.func(self.input), f"Should be {self.exp_output}"
