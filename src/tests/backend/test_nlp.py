import unittest
from src.nlp.nlp import text_to_sql

class NLPTestCase(unittest.TestCase):
    def test_text_to_sql(self):
        result = text_to_sql('test')
        self.assertEqual(result, "SELECT * FROM dummy_table WHERE column LIKE '%test%'")

if __name__ == '__main__':
    unittest.main()
