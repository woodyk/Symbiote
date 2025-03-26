#!/usr/bin/env python3
# custom_compression_verbose.py

"""
Custom Compression Library with Verbose Output

This module provides two compression algorithms:
1. Enhanced LZ77 Compression Algorithm
2. Optimized Burrows-Wheeler Transform (BWT) with Move-to-Front Encoding (MTF),
   Run-Length Encoding (RLE), and Canonical Huffman Coding

Author: [Your Name]
Date: [Current Date]
"""

import heapq
from collections import deque
import math

# ------------------------------
# Enhanced LZ77 Compression
# ------------------------------

def lz77_compress(data, verbose=False):
    """
    Compresses the input data using an enhanced LZ77 algorithm.
    """
    window_size = 4096  # Initial window size
    lookahead_buffer_size = 15  # Max match length
    i = 0  # Current position in data
    compressed_data = []

    while i < len(data):
        match = lz77_find_longest_match(data, i, window_size, lookahead_buffer_size)
        if match:
            (best_match_distance, best_match_length) = match
            next_index = i + best_match_length
            next_symbol = data[next_index] if next_index < len(data) else ''
            compressed_data.append((best_match_distance, best_match_length, next_symbol))
            if verbose:
                print(f"Match found: distance={best_match_distance}, length={best_match_length}, next_symbol='{next_symbol}'")
            i += best_match_length + 1
        else:
            compressed_data.append((0, 0, data[i]))
            if verbose:
                print(f"No match found: symbol='{data[i]}'")
            i += 1
    encoded_data = lz77_encode_compressed_data(compressed_data)
    if verbose:
        print(f"Compressed data length: {len(encoded_data)} bytes")
    return encoded_data

def lz77_find_longest_match(data, current_position, window_size, max_match_length):
    """
    Finds the longest match to a substring starting at the current_position.
    """
    end_of_buffer = min(current_position + max_match_length, len(data))
    best_match_distance = -1
    best_match_length = -1
    search_start = max(0, current_position - window_size)
    search_buffer = data[search_start:current_position]

    for match_length in range(1, end_of_buffer - current_position + 1):
        substring = data[current_position:current_position + match_length]
        index = search_buffer.rfind(substring)
        if index != -1:
            best_match_distance = current_position - (search_start + index)
            best_match_length = match_length
        else:
            break
    if best_match_distance > 0 and best_match_length > 0:
        return (best_match_distance, best_match_length)
    else:
        return None

def lz77_encode_compressed_data(compressed_data):
    """
    Encodes the compressed data into a byte string using variable-length codes.
    """
    encoded_data = bytearray()
    for item in compressed_data:
        distance, length, symbol = item
        encoded_data.extend(encode_varint(distance))
        encoded_data.extend(encode_varint(length))
        if symbol:
            encoded_data.append(ord(symbol))
    return bytes(encoded_data)

def lz77_decompress(encoded_data, verbose=False):
    """
    Decompresses data compressed with the enhanced LZ77 algorithm.
    """
    decompressed_data = []
    index = 0
    data_len = len(encoded_data)
    while index < data_len:
        distance, index = decode_varint(encoded_data, index)
        length, index = decode_varint(encoded_data, index)
        symbol = ''
        if index < data_len:
            symbol = chr(encoded_data[index])
            index += 1
        if distance == 0 and length == 0:
            decompressed_data.append(symbol)
            if verbose:
                print(f"Literal added: '{symbol}'")
        else:
            start = len(decompressed_data) - distance
            for i in range(length):
                decompressed_data.append(decompressed_data[start + i])
            if symbol:
                decompressed_data.append(symbol)
            if verbose:
                print(f"Copied {length} symbols from distance {distance}, appended symbol '{symbol}'")
    return ''.join(decompressed_data)

# Variable-length integer encoding functions
def encode_varint(number):
    """
    Encodes an integer into a variable-length format.
    """
    bytes_out = bytearray()
    while True:
        to_write = number & 0x7F
        number >>= 7
        if number:
            bytes_out.append(to_write | 0x80)
        else:
            bytes_out.append(to_write)
            break
    return bytes_out

def decode_varint(data, index):
    """
    Decodes a variable-length integer from data starting at index.
    """
    number = 0
    shift = 0
    while True:
        if index >= len(data):
            raise ValueError("Incomplete varint.")
        byte = data[index]
        index += 1
        number |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return number, index

# ------------------------------
# Optimized BWT-MTF-RLE-Huffman Compression
# ------------------------------

def bwt_transform_sa(s, verbose=False):
    """
    Performs the Burrows-Wheeler Transform using a Suffix Array.
    """
    s += '\0'  # Append end-of-text marker
    n = len(s)
    # Generate suffixes and their indices
    suffixes = [(s[i:], i) for i in range(n)]
    # Sort the suffixes
    suffixes.sort()
    # Build the BWT result
    bwt = [s[(idx - 1) % n] for _, idx in suffixes]
    if verbose:
        print(f"BWT Transform complete. Length: {len(bwt)}")
    return ''.join(bwt)

def bwt_inverse_optimized(r, verbose=False):
    """
    Reverses the Burrows-Wheeler Transform using LF-mapping.
    """
    n = len(r)
    count = {}
    total = {}
    for c in r:
        count[c] = count.get(c, 0) + 1

    sorted_chars = sorted(count.keys())
    total_count = 0
    for c in sorted_chars:
        total[c] = total_count
        total_count += count[c]

    ranks = []
    temp_count = {}
    for c in r:
        temp_count[c] = temp_count.get(c, 0) + 1
        ranks.append(temp_count[c])

    lf_mapping = [0] * n
    for i in range(n):
        c = r[i]
        lf_mapping[i] = total[c] + ranks[i] - 1

    idx = r.index('\0')
    result = []
    for _ in range(n - 1):
        idx = lf_mapping[idx]
        result.append(r[idx])

    restored = ''.join(reversed(result))
    if verbose:
        print(f"Inverse BWT complete. Restored data length: {len(restored)}")
    return restored

def mtf_encode_optimized(data, verbose=False):
    """
    Optimized MTF encoding using a deque.
    """
    symbols = deque([chr(i) for i in range(256)])
    result = []
    for char in data:
        index = symbols.index(char)
        result.append(index)
        symbols.remove(char)
        symbols.appendleft(char)
    if verbose:
        print(f"MTF Encoding complete. Length: {len(result)}")
    return result

def mtf_decode_optimized(indices, verbose=False):
    """
    Optimized MTF decoding using a deque.
    """
    symbols = deque([chr(i) for i in range(256)])
    result = []
    for index in indices:
        char = symbols[index]
        result.append(char)
        symbols.remove(char)
        symbols.appendleft(char)
    decoded = ''.join(result)
    if verbose:
        print(f"MTF Decoding complete. Length: {len(decoded)}")
    return decoded

def rle_encode(data, verbose=False):
    """
    Performs Run-Length Encoding on the input data.
    """
    encoded = []
    if not data:
        return encoded
    prev_symbol = data[0]
    count = 1
    for symbol in data[1:]:
        if symbol == prev_symbol and count < 255:
            count += 1
        else:
            encoded.append((prev_symbol, count))
            prev_symbol = symbol
            count = 1
    encoded.append((prev_symbol, count))
    if verbose:
        print(f"RLE Encoding complete. Length: {len(encoded)}")
    return encoded

def rle_decode(data, verbose=False):
    """
    Decodes data encoded with Run-Length Encoding.
    """
    decoded = []
    for symbol, count in data:
        decoded.extend([symbol] * count)
    if verbose:
        print(f"RLE Decoding complete. Length: {len(decoded)}")
    return decoded

# Huffman coding classes and functions
class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    # Define comparison operators for heapq
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [HuffmanNode(symbol=symbol, freq=freq) for symbol, freq in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq, left=node1, right=node2)
        heapq.heappush(heap, merged)
    return heap[0] if heap else None

def generate_huffman_codes(node, prefix='', codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        generate_huffman_codes(node.left, prefix + '0', codebook)
        generate_huffman_codes(node.right, prefix + '1', codebook)
    return codebook

def build_canonical_codes(codebook):
    """
    Builds canonical Huffman codes from the given codebook.
    """
    # Get list of symbols and their code lengths
    lengths = {symbol: len(code) for symbol, code in codebook.items()}
    # Sort symbols by code length and symbol value
    sorted_symbols = sorted(lengths.items(), key=lambda item: (item[1], item[0]))
    # Assign canonical codes
    canonical_codes = {}
    code = 0
    prev_length = 0
    for symbol, length in sorted_symbols:
        code <<= (length - prev_length)
        canonical_codes[symbol] = f'{code:0{length}b}'
        code += 1
        prev_length = length
    return canonical_codes

def huffman_encode_canonical(data, codebook):
    return ''.join([codebook[symbol] for symbol in data])

def huffman_decode_canonical(encoded_data, codebook):
    reverse_codebook = {v: k for k, v in codebook.items()}
    decoded_output = []
    code = ''
    for bit in encoded_data:
        code += bit
        if code in reverse_codebook:
            decoded_output.append(reverse_codebook[code])
            code = ''
    return decoded_output  # Return list of symbols

# Compression and decompression functions with RLE
def bwt_mtf_rle_huffman_compress(data, verbose=False):
    """
    Compresses the input data using BWT, MTF, RLE, and Canonical Huffman coding.
    """
    # BWT Transform using Suffix Array
    bwt_data = bwt_transform_sa(data, verbose)
    # MTF Encoding using deque
    mtf_data = mtf_encode_optimized(bwt_data, verbose)
    # RLE Encoding
    rle_data = rle_encode(mtf_data, verbose)
    # Flatten RLE data into symbols
    rle_symbols = []
    for symbol, count in rle_data:
        rle_symbols.append(symbol)
        rle_symbols.append(count)
    if verbose:
        print(f"RLE Symbols Length: {len(rle_symbols)}")
    # Build frequency table
    frequency = {}
    for symbol in rle_symbols:
        frequency[symbol] = frequency.get(symbol, 0) + 1
    if verbose:
        print(f"Frequency Table: {frequency}")
    # Build Huffman Tree
    huffman_root = build_huffman_tree(frequency)
    # Generate Huffman Codes
    original_codebook = generate_huffman_codes(huffman_root)
    # Generate Canonical Huffman Codes
    canonical_codebook = build_canonical_codes(original_codebook)
    # Huffman Encoding using Canonical Codes
    huffman_encoded = huffman_encode_canonical(rle_symbols, canonical_codebook)
    if verbose:
        compressed_size_bits = len(huffman_encoded)
        compressed_size_bytes = math.ceil(compressed_size_bits / 8)
        print(f"Huffman Encoded Length: {compressed_size_bits} bits ({compressed_size_bytes} bytes)")
    return huffman_encoded, canonical_codebook

def bwt_mtf_rle_huffman_decompress(huffman_encoded, canonical_codebook, verbose=False):
    """
    Decompresses data compressed with BWT, MTF, RLE, and Canonical Huffman coding.
    """
    # Huffman Decoding using Canonical Codes
    huffman_decoded = huffman_decode_canonical(huffman_encoded, canonical_codebook)
    if verbose:
        print(f"Huffman Decoded Length: {len(huffman_decoded)}")
    # Reconstruct RLE data
    rle_data = []
    iterator = iter(huffman_decoded)
    for symbol in iterator:
        count = next(iterator)
        rle_data.append((symbol, count))
    if verbose:
        print(f"Reconstructed RLE Data Length: {len(rle_data)}")
    # RLE Decoding
    mtf_decoded_indices = rle_decode(rle_data, verbose)
    # MTF Decoding using deque
    mtf_decoded = mtf_decode_optimized(mtf_decoded_indices, verbose)
    # Inverse BWT using LF-mapping
    restored_data = bwt_inverse_optimized(mtf_decoded, verbose)
    return restored_data

# ------------------------------
# Utility Functions
# ------------------------------

def calculate_compression_ratio(original_data, compressed_data):
    """
    Calculates the compression ratio.
    """
    original_size = len(original_data.encode('utf-8'))
    if isinstance(compressed_data, bytes):
        compressed_size = len(compressed_data)
    else:
        # Convert bits to bytes, rounding up for partial bytes
        compressed_size = math.ceil(len(compressed_data) / 8)
    ratio = (1 - (compressed_size / original_size)) * 100
    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {compressed_size} bytes")
    return ratio

# ------------------------------
# Main Testing Block
# ------------------------------

if __name__ == "__main__":
    # Test data: Highly repetitive data
    test_string_repetitive = "The quick brown fox jumps over the lazy dog. " * 1000  # Larger dataset

    # Test data: Less repetitive data (Alice's Adventures in Wonderland excerpt)
    test_string_non_repetitive = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do: once or twice she had peeped into the book her "
        "sister was reading, but it had no pictures or conversations in it, 'and what "
        "is the use of a book,' thought Alice 'without pictures or conversation?'"
    )

    # Enhanced LZ77 Compression Test on Highly Repetitive Data
    print("Testing Enhanced LZ77 Compression on Highly Repetitive Data...")
    lz77_compressed = lz77_compress(test_string_repetitive, verbose=False)
    lz77_decompressed = lz77_decompress(lz77_compressed, verbose=False)
    assert lz77_decompressed == test_string_repetitive, "LZ77 decompression failed on repetitive data!"
    lz77_ratio = calculate_compression_ratio(test_string_repetitive, lz77_compressed)
    print(f"LZ77 Compression Ratio on Repetitive Data: {lz77_ratio:.2f}%\n")

    # BWT-MTF-RLE-Huffman Compression Test on Highly Repetitive Data
    print("Testing Optimized BWT-MTF-RLE-Huffman Compression on Highly Repetitive Data...")
    bwt_compressed_opt, canonical_codebook = bwt_mtf_rle_huffman_compress(test_string_repetitive, verbose=False)
    bwt_decompressed_opt = bwt_mtf_rle_huffman_decompress(bwt_compressed_opt, canonical_codebook, verbose=False)
    assert bwt_decompressed_opt == test_string_repetitive, "Optimized BWT-MTF-RLE-Huffman decompression failed on repetitive data!"
    bwt_ratio_opt = calculate_compression_ratio(test_string_repetitive, bwt_compressed_opt)
    print(f"Optimized BWT-MTF-RLE-Huffman Compression Ratio on Repetitive Data: {bwt_ratio_opt:.2f}%\n")

    # Enhanced LZ77 Compression Test on Less Repetitive Data
    print("Testing Enhanced LZ77 Compression on Less Repetitive Data...")
    lz77_compressed_non_rep = lz77_compress(test_string_non_repetitive, verbose=False)
    lz77_decompressed_non_rep = lz77_decompress(lz77_compressed_non_rep, verbose=False)
    assert lz77_decompressed_non_rep == test_string_non_repetitive, "LZ77 decompression failed on non-repetitive data!"
    lz77_ratio_non_rep = calculate_compression_ratio(test_string_non_repetitive, lz77_compressed_non_rep)
    print(f"LZ77 Compression Ratio on Non-Repetitive Data: {lz77_ratio_non_rep:.2f}%\n")

    # BWT-MTF-RLE-Huffman Compression Test on Less Repetitive Data
    print("Testing Optimized BWT-MTF-RLE-Huffman Compression on Less Repetitive Data...")
    bwt_compressed_opt_non_rep, canonical_codebook_non_rep = bwt_mtf_rle_huffman_compress(test_string_non_repetitive, verbose=False)
    bwt_decompressed_opt_non_rep = bwt_mtf_rle_huffman_decompress(bwt_compressed_opt_non_rep, canonical_codebook_non_rep, verbose=False)
    assert bwt_decompressed_opt_non_rep == test_string_non_repetitive, "Optimized BWT-MTF-RLE-Huffman decompression failed on non-repetitive data!"
    bwt_ratio_opt_non_rep = calculate_compression_ratio(test_string_non_repetitive, bwt_compressed_opt_non_rep)
    print(f"Optimized BWT-MTF-RLE-Huffman Compression Ratio on Non-Repetitive Data: {bwt_ratio_opt_non_rep:.2f}%")
