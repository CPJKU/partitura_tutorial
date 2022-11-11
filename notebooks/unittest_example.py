import unittest

# Import Predefined Test files
from tests import TEST_FILES

class TypicalTest(unittest.TestCase):
    """
    Typical Test File
    """
    def test_something(self):
        document_path = TEST_FILES[0]
        function = lambda x : 1
        result = 1
        self.assertTrue(function(document_path) == result, "Incorect Result")