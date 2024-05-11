from test import OutputTest

class PosEncTest:

    def __init__(self):
        self.input = True
        self.exp_output = True
        self.func = True
        self.test = OutputTest(self.input, self.exp_output, self.func)

    def __call__(self):
        self.test()
