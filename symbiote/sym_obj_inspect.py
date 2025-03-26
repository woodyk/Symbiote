#!/usr/bin/env python3
#
# symbiote/sym_obj_inspect.py

import sys
from rich.pretty import pprint as print 
import re

def max_depth(obj, current_depth=0):
    """
    Determine the maximum depth of a nested dictionary or list.

    Args:
        obj: The object to check (dict, list, or other).
        current_depth: Tracks the current depth in recursion.

    Returns:
        int: The maximum depth of the object.
    """
    if isinstance(obj, dict):
        return max((max_depth(v, current_depth + 1) for v in obj.values()), default=current_depth)
    elif isinstance(obj, list):
        return max((max_depth(v, current_depth + 1) for v in obj), default=current_depth)
    else:
        return current_depth

def obj_stats(container):
    """
    Gather statistics about a container (dict, list, etc.).

    Args:
        container: The container object to analyze.

    Returns:
        dict: A summary of statistics about the container.
    """
    stats = {
        'type': type(container).__name__,
        'length': len(container) if hasattr(container, '__len__') else 'N/A',
        'memory_size': sys.getsizeof(container),
        'is_iterable': hasattr(container, '__iter__'),
        'max_depth': max_depth(container) if isinstance(container, (dict, list)) else 'N/A',
        'first_5_elements': list(container)[:5] if hasattr(container, '__iter__') else 'N/A',
    }
    return stats

def obj_search(obj, pattern, n=20, path='', snippets=None):
    """
    Search for a keyword or regex pattern within a container object.

    Args:
        obj: The container to search (dict, list, str, etc.).
        pattern: The keyword or regex pattern to search for.
        n: The number of characters before and after the match in the snippet.
        path: The current path to the object (used for recursion).
        snippets: A dictionary to store matched snippets (used for recursion).

    Returns:
        tuple: A tuple containing:
            - A list of paths to matched objects.
            - A dictionary of snippets (path -> snippet).
    """
    if snippets is None:
        snippets = {}

    matches = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            subpath = f"{path}.{key}" if path else key
            submatches, sub_snippets = obj_search(value, pattern, n, subpath, snippets)
            matches.extend(submatches)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            subpath = f"{path}[{index}]"
            submatches, sub_snippets = obj_search(value, pattern, n, subpath, snippets)
            matches.extend(submatches)
    elif isinstance(obj, str):
        for match in re.finditer(pattern, obj):
            start, end = match.start(), match.end()
            snippet = obj[max(0, start - n): min(len(obj), end + n)]
            matches.append(path)
            snippets[path] = snippet
    elif isinstance(obj, (int, float)):
        if re.fullmatch(pattern, str(obj)):
            matches.append(path)
            snippets[path] = str(obj)

    return matches, snippets

# Test cases
if __name__ == "__main__":
    # Test 1: Simple nested dictionary
    nested_dict = {
        "a": {"b": {"c": {"d": 1}}},
        "e": {"f": 2},
    }
    print("Test 1: Nested Dictionary")
    print(obj_stats(nested_dict))  # Should include max_depth=4

    # Test 2: Nested list
    nested_list = [1, [2, [3, [4, [5]]]]]
    print("\nTest 2: Nested List")
    print(obj_stats(nested_list))  # Should include max_depth=5

    # Test 3: Flat dictionary
    flat_dict = {"a": 1, "b": 2, "c": 3}
    print("\nTest 3: Flat Dictionary")
    print(obj_stats(flat_dict))  # Should include max_depth=1

    # Test 4: Flat list
    flat_list = [1, 2, 3, 4, 5]
    print("\nTest 4: Flat List")
    print(obj_stats(flat_list))  # Should include max_depth=1

    # Test 5: Long string
    long_string = "a" * 1050
    print("\nTest 5: Long String")
    print(obj_stats(long_string))  # Should not include max_depth

    # Test 6: Empty dictionary
    empty_dict = {}
    print("\nTest 6: Empty Dictionary")
    print(obj_stats(empty_dict))  # Should include max_depth=0

    # Test 7: Empty list
    empty_list = []
    print("\nTest 7: Empty List")
    print(obj_stats(empty_list))  # Should include max_depth=0

    # Test 8: Mixed container
    mixed_container = {
        "key1": [1, 2, {"nested_key": [3, 4, {"deep_key": 5}]}],
        "key2": {"another_key": [6, 7]},
    }
    print("\nTest 8: Mixed Container")
    print(obj_stats(mixed_container))  # Should include max_depth=4

    nested_dict = {
        "key1": "This is a test string.",
        "key2": {
            "subkey1": "Another test string with keyword test.",
            "subkey2": [1, 2, "Test again!"]
        },
        "key3": ["No match here", "test test test"],
        "key4": 12345
    }

    # Test search functionality
    print("\nTest Search Function")
    matches, snippets = obj_search(nested_dict, r"test", n=10)
    print(f"Matches:")
    print(matches)
    print("Snippets:")
    print(snippets)
