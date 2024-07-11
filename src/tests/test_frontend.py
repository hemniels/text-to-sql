import unittest
from src.nlp.app import app

class BackendTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_sql_query(self):
        response = self.app.post('/api/sql-query', json={'query': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('SELECT * FROM dummy_table WHERE column LIKE \'%test%\'', response.get_data(as_text=True))

if __name__ == '__main__':
    unittest.main()
