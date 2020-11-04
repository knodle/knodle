import unittest

from test_script import calculate_average


class TestAvg(unittest.TestCase):
    def test_list_int(self):
        """ Test that an average f input integers is calculated correctly """
        data = [1, 2, 3]
        result = calculate_average(data)
        self.assertEqual(result, 2)


if __name__ == '__main__':
    unittest.main()

