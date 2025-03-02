#!/usr/bin/env python3

class CustomOperations:
    def __init__(self, symbiote_instance):
        self.symbiote_instance = symbiote_instance
        self.operations = {}

    def add_operation(self, name, operation):
        """
        Add a new custom operation.

        :param name: The name of the operation.
        :param operation: The operation, which should be a function.
        """
        self.operations[name] = operation

    def remove_operation(self, name):
        """
        Remove a custom operation.

        :param name: The name of the operation.
        """
        if name in self.operations:
            del self.operations[name]

    def execute_operation(self, name, *args, **kwargs):
        """
        Execute a custom operation.

        :param name: The name of the operation.
        :param args: Positional arguments for the operation.
        :param kwargs: Keyword arguments for the operation.
        """
        if name in self.operations:
            return self.operations[name](self.symbiote_instance, *args, **kwargs)
        else:
            raise Exception(f"No operation named '{name}' found.")

    def save_operations(self):
        """
        Save the current state of custom operations to a file.
        """
        with open('custom_operations.pkl', 'wb') as f:
            pickle.dump(self.operations, f)

    def load_operations(self):
        """
        Load custom operations from a file.
        """
        with open('custom_operations.pkl', 'rb') as f:
            self.operations = pickle.load(f)

