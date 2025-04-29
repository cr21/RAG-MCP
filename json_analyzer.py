import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Any
from pprint import pprint

class JsonStructureAnalyzer:
    def __init__(self):
        self.unique_keys = defaultdict(set)
        self.unique_nested_keys = {
            'articleAttributes': set(),
            'masterCategory': set(),
            'subCategory': set(),
            'articleType': set(),
            'productDescriptors': set()
        }
        self.data_types = defaultdict(set)
        self.value_samples = defaultdict(set)
        self.total_files = 0
        self.processed_files = 0
        self.error_files = 0

    def analyze_value(self, key: str, value: Any):
        """Analyze the data type and sample values for a key"""
        self.data_types[key].add(type(value).__name__)
        
        # Store sample values (limit to 5 samples per key)
        if len(self.value_samples[key]) < 5:
            if isinstance(value, (str, int, float, bool)):
                self.value_samples[key].add(str(value))

    def analyze_nested_structure(self, data: Dict, parent_key: str = None):
        """Recursively analyze nested structure"""
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                self.unique_keys[full_key].add(type(value).__name__)
                
                # Analyze specific nested structures we're interested in
                if key in self.unique_nested_keys:
                    if isinstance(value, dict):
                        self.unique_nested_keys[key].update(value.keys())
                
                self.analyze_value(full_key, value)
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    self.analyze_nested_structure(value, full_key)
        elif isinstance(data, list):
            for item in data:
                self.analyze_nested_structure(item, parent_key)

    def process_file(self, file_path: Path):
        """Process a single JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "data" in data:
                    self.analyze_nested_structure(data["data"])
                    self.processed_files += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            self.error_files += 1

    def analyze_directory(self, directory_path: str):
        """Analyze all JSON files in the directory"""
        directory = Path(directory_path)
        json_files = list(directory.glob("*.json"))
        self.total_files = len(json_files)
        
        print(f"Found {self.total_files} JSON files to process...")
        
        for i, file_path in enumerate(json_files, 1):
            if i % 1000 == 0:
                print(f"Processed {i}/{self.total_files} files...")
            self.process_file(file_path)

    def print_analysis(self):
        """Print the analysis results"""
        print("\n=== Analysis Results ===")
        print(f"Total files: {self.total_files}")
        print(f"Successfully processed: {self.processed_files}")
        print(f"Errors encountered: {self.error_files}")
        
        print("\n=== Essential Categories Analysis ===")
        for category in self.unique_nested_keys:
            print(f"\n{category} unique keys:")
            print(f"Total unique keys: {len(self.unique_nested_keys[category])}")
            pprint(sorted(self.unique_nested_keys[category]))
        
        print("\n=== Basic Fields Analysis ===")
        basic_fields = [
            'id', 'price', 'discountedPrice', 'styleType', 'productTypeId',
            'articleNumber', 'visualTag', 'productDisplayName', 'variantName',
            'myntraRating', 'catalogAddDate', 'brandName', 'ageGroup', 'gender',
            'baseColour', 'colour1', 'colour2', 'fashionType', 'season', 'year',
            'usage', 'vat', 'displayCategories', 'weight'
        ]
        
        for field in basic_fields:
            if field in self.data_types:
                print(f"\n{field}:")
                print(f"Data types: {self.data_types[field]}")
                print(f"Sample values: {self.value_samples[field]}")

def main():
    analyzer = JsonStructureAnalyzer()
    directory_path = "/Users/chiragtagadiya/Downloads/product_new/fashion-dataset/styles"
    
    print("Starting JSON structure analysis...")
    analyzer.analyze_directory(directory_path)
    analyzer.print_analysis()

if __name__ == "__main__":
    main()
