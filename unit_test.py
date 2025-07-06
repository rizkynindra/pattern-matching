import unittest
from flask import json
from main import app


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_validate_phone_number_valid(self):
        response = self.app.post('/number', json={'number': '+6281234567890', 'region': 'ID'})
        data = json.loads(response.data)
        self.assertIn('valid', data)
        self.assertTrue(data['valid'])
        self.assertEqual(response.status_code, 200)

    def test_validate_phone_number_invalid(self):
        response = self.app.post('/number', json={'number': 'abcd1234'})
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(response.status_code, 200)

    def test_validate_email_valid(self):
        response = self.app.post('/email', json={'email': 'test@gmail.com', 'name': 'Test'})
        data = json.loads(response.data)
        self.assertIn('valid_syntax', data)
        self.assertTrue(data['valid_syntax'])
        self.assertEqual(response.status_code, 200)

    def test_validate_email_invalid(self):
        response = self.app.post('/email', json={'email': 'invalid-email', 'name': 'Test User'})
        data = json.loads(response.data)
        self.assertIn('valid_syntax', data)
        self.assertFalse(data['valid_syntax'])
        self.assertEqual(response.status_code, 200)

    def test_validate_missing_email_name(self):
        response = self.app.post('/email', json={'email': 'test@example.com'})
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(response.status_code, 400)

    def test_logging_function(self):
        response = self.app.post('/number', json={'number': '+6281234567890'})
        self.assertEqual(response.status_code, 200)

    def test_image_matching_valid(self):
        response = self.app.post('/image', json={'klaim': ['/1029150/17717/i/380/depositphotos_177176800-stock-photo-malay-teenage-doing-selfing-his.jpg'],
                                                 'pengkinian': ['/1029150/17717/i/380/depositphotos_177176800-stock-photo-malay-teenage-doing-selfing-his.jpg']})
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertEqual(response.status_code, 200)

    def test_image_matching_invalid(self):
        response = self.app.post('/image', json={'klaim': [], 'pengkinian': []})
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
