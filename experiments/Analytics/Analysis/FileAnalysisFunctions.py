#!/usr/bin/env python3
#
# FileAnalysisFunctions.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from utils import create_compact_cnn, prepare_data, train_model, evaluate_model, \
                  extract_metadata, extract_byte_sequence_statistics, extract_header_footer, \
                  extract_entropy, extract_hashes, extract_strings

class FileAnomalyDetection:
    def __init__(self, model_path='./model/malware_detection.h5'):
        self.model_path = model_path
        self.model = load_model(model_path)

    def _extract_features(self, file_path):
        """Extract features from the given file."""

        metadata = extract_metadata(file_path)
        byte_seq_stats = extract_byte_sequence_statistics(file_path)
        header_footer = extract_header_footer(file_path)
        entropy = extract_entropy(file_path)
        hashes = extract_hashes(file_path)
        strings = extract_strings(file_path)

        features = np.concatenate((metadata, byte_seq_stats, header_footer, entropy, hashes, strings))

        return features

    def _predict_file(self, file_path):
        """Predict whether the given file is clean or infected."""

        features = self._extract_features(file_path)
        pred = self.model.predict(np.expand_dims(features, axis=0)).round()

        return pred

    def evaluate_model(self, x_test, y_test):
        """Evaluate the model on the test set."""

        evaluate_model(self.model, x_test, y_test)

    def analyze_files(self, file_paths, print_summary=True):
        """Analyze the given list of files."""

        for idx, fp in enumerate(file_paths):
            try:
                prediction = self._predict_file(fp)[0][0]
                label = 'Clean' if prediction < 0.5 else 'Malware'
                message = f"File [{idx}] '{fp}' classified as {label}"

                if print_summary:
                    print(message)

                if label == 'Malware':
                    print(classification_report(np.zeros((1)), np.ones((1))))

            except Exception as err:
                print(f"Couldn't analyze '{fp}', reason: {err}")


class FileAnalysisFunctions:

    @staticmethod
    def extract_metadata(file_path):
        """Extract file metadata"""
        return {'file_size': os.path.getsize(file_path),
                'last_modified': datetime.datetime.fromtimestamp(os.path.getctime(file_path))}

    @staticmethod
    def extract_byte_sequence_statistics(file_path):
        """Calculate byte sequence statistics"""
        raw_data = open(file_path, 'rb').read()
        byte_freq = Counter(raw_data)
        min_count = max(len(byte_freq), 10)
        sorted_byte_freq = sorted(((key, value) for key, value in byte_freq.items()),
                                 key=lambda pair: (-pair[1], pair[0]))[:min_count]
        top_chars = ['{}:{}({})'.format(*item) for item in sorted_byte_freq]
        return ', '.join(top_chars)

    @staticmethod
    def extract_header_footer(file_path):
        """Check for anomalies in header and footer"""
        header_size = 256
        footer_size = 256
        raw_data = open(file_path, 'rb').read()
        header = raw_data[:header_size]
        footer = raw_data[-footer_size:]
        header_match = re.search(r'\x00.*\xFF|\xFF.*\x00', header)
        footer_match = re.search(r'\x00.*\xFF|\xFF.*\x00', footer)
        if header_match:
            header_result = 'Header possibly contains null-padding issue.'
        elif len(header) > 0 and chr(header[0]).encode().isalpha():
            header_result = 'Header does not seem to start with SOH.'
        else:
            header_result = 'No apparent issues in header.'
        if footer_match:
            footer_result = 'Footer possibly contains null-padding issue.'
        elif len(footer) > 0 and chr(footer[-1]).encode().isalpha():
            footer_result = 'Footer does not seem to end with EOT.'
        else:
            footer_result = 'No apparent issues in footer.'
        return {'header_result': header_result, 'footer_result': footer_result}

    @staticmethod
    def extract_entropy(file_path):
        """Calculate file entropy"""
        raw_data = open(file_path, 'rb').read()
        entropy = []
        for i in range(len(raw_data) - 511):
            window = raw_data[i:i + 512]
            entropy.append(-sum(prob * math.log(prob, 2) for prob in
                              Counter(window).values()) / 8)
        return np.mean(entropy)

    @staticmethod
    def extract_hashes(file_path):
        """Compute SHA-256 and SHA-512 hashes"""
        sha256_hash = hashlib.sha256()
        sha512_hash = hashlib.sha512()
        BUFFER_SIZE = 65536
        with open(file_path, 'rb') as f:
            buffer = f.read(BUFFER_SIZE)
            while len(buffer) > 0:
                sha256_hash.update(buffer)
                sha512_hash.update(buffer)
                buffer = f.read(BUFFER_SIZE)
        return {'sha256': sha256_hash.hexdigest(), 'sha512': sha512_hash.hexdigest()}

    @staticmethod
    def extract_strings(file_path):
        """Search for strings in the file content"""
        raw_data = open(file_path, 'rb').read()
        strings = re.findall(r'(?:\x00)*(?:[\w\-_\.\(\)\{\}\|\\/\+\*\^\"\';<>]+)(?:\x00)+', raw_data.decode())
        return ', '.join(sorted(set(strings)))

    @staticmethod
    def check_text_binary_anomaly(file_path):
        """Verify if the file contains text or binary content"""
        raw_data = open(file_path, 'rb').read()
        if raw_data.startswith(b'\x00'):
            return 'Binary zero byte at the very beginning of the file!'
        elif raw_data.endswith(b'\x00'):
            return 'Binary zero byte at the very end of the file!'
        elif re.search(br'[\x00]*[\x01-\x08]+[\x00]*', raw_data):
            return 'Possibly invalid text encoding.'
        elif re.search(br'[\x00]*[\x0B-\x1F]+[\x00]*', raw_data):
            return 'Non-printable ASCII characters found.'
        else:
            return ''

    @staticmethod
    def check_audio_anomalies(file_path):
        """Detect anomalies in audio files"""
        audio_tags = ['RIFF', 'WAVE', 'fmt ', 'data']
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            for tag in audio_tags:
                if tag not in str(raw_data)[:10]:
                    return 'Invalid tag {}, probably corrupt file.'.format(tag)
        return ''

    @staticmethod
    def check_image_anomalies(file_path):
        """Detect anomalies in image files"""
        img_tags = ['BM', 'GIF', 'PNG', 'TIFF', 'JFIF']
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            for tag in img_tags:
                if tag not in str(raw_data)[:10]:
                    return 'Invalid tag {}, probably corrupt file.'.format(tag)
        return ''

    @staticmethod
    def extract_file_network_connections(file_path):
        """Scan for network connections initiated by the file"""
        connection_found = False
        for line in subprocess.run(['lsof', '-p', str(os.getpid())], stdout=subprocess.PIPE).stdout.split(b'\n'):
            pid, cmd, fd, typ, dev, ino, maj, min, nlink, uid, gid, rdev, blksize, blkno, size, node, path = map(str, line.strip().split()[1:-1])
            if 'virusescan' not in cmd and 'python' in cmd and file_path in path:
                connection_found = True
                print('Network Connection Found: {}'.format(line.strip()))
        if not connection_found:
            print('No network connections found.')

    @staticmethod
    def extract_process_behavior_anomalies(file_path):
        """Monitor process behavior while running the file"""
        temp_file = 'temp_{}.txt'.format(uuid.uuid4())
        subprocess.call(['cp', file_path, temp_file])
        proc = subprocess.Popen(['virusescan', temp_file], bufsize=-1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.decode().strip())
        os.remove(temp_file)
        if '[INFO ]' in lines[-1]:
            print('Process finished successfully.')
        else:
            print('Anomalous behavior detected.')

class FileAnalysisAndAnomalyDetection:
    """Wadih Frederick Khairallah's File Analysis And Anomaly Detection Class"""

    def __init__(self):
        self.watermark = "# Created by Wadih Frederick Khairallah, Copyright © 2023"

    @staticmethod
    def extract_metadata(file_path):
        """Extract file metadata"""
        return extract_metadata(file_path)

    @staticmethod
    def extract_byte_sequence_statistics(file_path):
        """Calculate byte sequence statistics"""
        return extract_byte_sequence_statistics(file_path)

    @staticmethod
    def extract_header_footer(file_path):
        """Check for anomalies in header and footer"""
        return extract_header_footer(file_path)

    @staticmethod
    def extract_entropy(file_path):
        """Calculate file entropy"""
        return extract_entropy(file_path)

    @staticmethod
    def extract_hashes(file_path):
        """Compute SHA-256 and SHA-512 hashes"""
        return extract_hashes(file_path)

    @staticmethod
    def extract_strings(file_path):
        """Search for strings in the file content"""
        return extract_strings(file_path)

    @staticmethod
    def check_text_binary_anomaly(file_path):
        """Verify if the file contains text or binary content"""
        return check_text_binary_anomaly(file_path)

    @staticmethod
    def check_audio_anomalies(file_path):
        """Detect anomalies in audio files"""
        return check_audio_anomalies(file_path)

    @staticmethod
    def check_image_anomalies(file_path):
        """Detect anomalies in image files"""
        return check_image_anomalies(file_path)

    @staticmethod
    def extract_file_network_connections(file_path):
        """Scan for network connections initiated by the file"""
        return extract_file_network_connections(file_path)

    @staticmethod
    def extract_process_behavior_anomalies(file_path):
        """Monitor process behavior while running the file"""
        return extract_process_behavior_anomalies(file_path)

    @staticmethod
    def extract_structural_anomalies(file_path):
        """Chain of anomaly detection functions"""
        metadata = FileAnalysisAndAnomalyDetection.extract_metadata(file_path)
        byte_sequence_stats = FileAnalysisAndAnomalyDetection.extract_byte_sequence_statistics(file_path)
        header_footer_anoms = FileAnalysisAndAnomalyDetection.extract_header_footer(file_path)
        entropy_anoms = FileAnalysisAndAnomalyDetection.extract_entropy(file_path)
        hash_anoms = FileAnalysisAndAnomalyDetection.extract_hashes(file_path)
        string_anoms = FileAnalysisAndAnomalyDetection.extract_strings(file_path)
        text_binary_anoms = FileAnalysisAndAnomalyDetection.check_text_binary_anomaly(file_path)
        audio_anoms = FileAnalysisAndAnomalyDetection.check_audio_anomalies(file_path)
        image_anoms = FileAnalysisAndAnomalyDetection.check_image_anomalies(file_path)
        netconn_anoms = FileAnalysisAndAnomalyDetection.extract_file_network_connections(file_path)
        process_anoms = FileAnalysisAndAnomalyDetection.extract_process_behavior_anomalies(file_path)

        return {
            'metadata': metadata,
            'byte_sequence_stats': byte_sequence_stats,
            'header_footer_anoms': header_footer_anoms,
            'entropy_anoms': entropy_anoms,
            'hash_anoms': hash_anoms,
            'string_anoms': string_anoms,
            'text_binary_anoms': text_binary_anoms,
            'audio_anoms': audio_anoms,
            'image_anoms': image_anoms,
            'netconn_anoms': netconn_anoms,
            'process_anoms': process_anoms,
        }

    @staticmethod
    def analyze_file(file_path):
        """Main function to analyze a single file"""
        file_info = FileAnalysisAndAnomalyDetection.extract_structural_anomalies(file_path)
        print("\nFILE ANALYSIS RESULTS FOR {}".format(file_path))
        print(file_info)


