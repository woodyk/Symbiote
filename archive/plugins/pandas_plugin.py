#!/usr/bin/env python3
#
# pandas_plugin.py

import os
import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.data = self.read_file(file_path)

    def read_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension.lower() == '.tsv' or file_extension.lower() == '.txt':
            data = pd.read_csv(file_path, sep='\t')
        elif file_extension.lower() == '.json':
            data = pd.read_json(file_path)
        elif file_extension.lower() == '.xlsx':
            data = pd.read_excel(file_path)
        elif file_extension.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return data

    def filter_rows(self, column_name, value):
        self.data = self.data[self.data[column_name] == value]

    def drop_column(self, column_name):
        self.data = self.data.drop(columns=[column_name])

    def rename_column(self, old_name, new_name):
        self.data = self.data.rename(columns={old_name: new_name})

    def sort_by(self, column_name, ascending=True):
        self.data = self.data.sort_values(by=column_name, ascending=ascending)

    def get_head(self, n=5):
        return self.data.head(n)

# Example usage
file_path = 'path/to/your/file'
processor = DataProcessor(file_path)

# Perform data manipulation
processor.filter_rows('column_name', 'value')
processor.drop_column('column_name')
processor.rename_column('old_name', 'new_name')
processor.sort_by('column_name', ascending=False)

# Print the first 5 rows of the processed data
print(processor.get_head())
