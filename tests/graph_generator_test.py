import os
import unittest
import torch
import shutil
from utils.graph_generator import generate_and_save_graphs, conditions

class TestGraphGenerator(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join("..", "data", "test_graphs")
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.num_graphs = 2  # Small number for testing

    def tearDown(self):
        shutil.rmtree(self.test_output_dir)

    def check_generated_graphs(self, condition_name):
        output_file = os.path.join(self.test_output_dir, f"{condition_name}_graphs.pt")
        self.assertTrue(os.path.exists(output_file), f"{output_file} does not exist")
        graphs = torch.load(output_file)
        self.assertEqual(len(graphs), self.num_graphs, f"Expected {self.num_graphs} graphs, but got {len(graphs)}")

    def generate_and_check(self, condition_name, params):
        generate_and_save_graphs(
            num_graphs=self.num_graphs,
            output_dir=self.test_output_dir,
            condition_name=condition_name,
            **params
        )
        self.check_generated_graphs(condition_name)

    def test_generate_and_save_graphs(self):
        for condition_name, params in conditions.items():
            self.generate_and_check(condition_name, params)

if __name__ == "__main__":
    unittest.main()