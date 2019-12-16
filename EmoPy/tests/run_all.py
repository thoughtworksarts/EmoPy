import unittest
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('.', pattern='test_*.py')
unittest.TextTestRunner(verbosity=2).run(test_suite)
