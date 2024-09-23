import unittest
import step4_classify_keys as da
import json
import os


class TestExtractImplicitSchemaPaths(unittest.TestCase):

    def setUp(self):
        self.schemas = {}

        for filename in os.listdir("test_schemas"):
            with open(os.path.join("test_schemas", filename), 'r') as f:
                self.schemas[filename] = json.load(f)
    
    def test_schema_1(self): # No additional properties are allowed
        schema = self.schemas["test1.json"]
        expected_paths = set()
        actual_paths = set(da.extract_implicit_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_2(self): # Additional properties is true for address
        schema = self.schemas["test2.json"]
        expected_paths = {("$", "address")}
        actual_paths = set(da.extract_implicit_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_3(self): # Additional properties is an object for address
        schema = self.schemas["test3.json"]
        expected_paths = {("$", "address"), ("$", "address", "**")}
        actual_paths = set(da.extract_implicit_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_4(self): # Additional properties is not declared for address
        schema = self.schemas["test4.json"]
        expected_paths = {("$", "address")}
        actual_paths = set(da.extract_implicit_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_5(self): # Additional properties are allowed at the top level
        schema = self.schemas["test5.json"]
        expected_paths = {("$",)}
        actual_paths = set(da.extract_implicit_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_6(self): # Pattern properties are used
        schema = self.schemas["test6.json"]
        expected_paths = [("$", "address")]
        actual_paths = set(da.extract_implicit_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    # Additional test cases for other schemas can be added here

class TestExtractStaticSchemaPaths(unittest.TestCase):

    def setUp(self):
        self.schemas = {}

        for filename in os.listdir("test_schemas"):
            with open(os.path.join("test_schemas", filename), 'r') as f:
                self.schemas[filename] = json.load(f)
    
    def test_schema_1(self): 
        schema = self.schemas["test1.json"]
        expected_paths = {("$", "address"), ("$", "person")}
        actual_paths = set(da.extract_static_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_2(self):
        schema = self.schemas["test2.json"]
        expected_paths = {("$", "address"), ("$", "person")}
        actual_paths = set(da.extract_static_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_3(self):
        schema = self.schemas["test3.json"]
        expected_paths = {("$", "address"), ("$", "address", "**"), ("$", "person")}
        actual_paths = set(da.extract_static_schema_paths(schema))
        print(actual_paths)
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_4(self):
        schema = self.schemas["test4.json"]
        expected_paths = {("$", "address"), ("$", "person")}
        actual_paths = set(da.extract_static_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    
    def test_schema_5(self):
        schema = self.schemas["test5.json"]
        expected_paths = {("$", "address"), ("$", "person")}
        actual_paths = set(da.extract_static_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    '''
    def test_schema_6(self):
        schema = self.schemas["test6.json"]
        expected_paths = [("$", "address")]
        actual_paths = set(da.extract_static_schema_paths(schema))
        self.assertCountEqual(actual_paths, expected_paths)
    '''

if __name__ == "__main__":
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestExtractStaticSchemaPaths))

