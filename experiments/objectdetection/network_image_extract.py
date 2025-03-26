#!/usr/bin/env python3
#
# network_image_extract.py

import threading
from scapy.all import sniff
from scapy.layers.http import HTTPRequest, HTTPResponse
from scapy.layers.inet import TCP
from PIL import Image
from io import BytesIO
import re
import base64

stop_sniffing = False

def extract_and_show_image(http_payload):
    try:
        # Look for content type that indicates an image
        content_type = re.search(b"Content-Type:\s*(image/\w+)", http_payload)
        if content_type:
            # Look for the image data
            image_data_start = re.search(b"\r\n\r\n", http_payload)
            if image_data_start:
                image_data = http_payload[image_data_start.end():]
                # Handle base64-encoded images (e.g., in data URLs)
                if b"Content-Transfer-Encoding: base64" in http_payload:
                    image_data = base64.b64decode(image_data)
                # Load the image data into a PIL image and display it
                image = Image.open(BytesIO(image_data))
                image.show()
    except Exception as e:
        print(f"Failed to extract and display image: {e}")

def process_packet(packet):
    global stop_sniffing
    if stop_sniffing:
        return True  # Stop sniffing if the global flag is set

    # Filter for HTTP traffic
    if packet.haslayer(HTTPRequest) or packet.haslayer(HTTPResponse):
        http_payload = bytes(packet[TCP].payload)
        extract_and_show_image(http_payload)

def stop_sniffing_on_keypress():
    global stop_sniffing
    print("Press 'q' to stop sniffing.")
    while not stop_sniffing:
        user_input = input().strip().lower()
        if user_input == 'q':
            print("Stopping sniffing...")
            stop_sniffing = True
            break

if __name__ == "__main__":
    # Replace 'eth0' with the correct interface name on your system
    interface = "en0"
    print(f"Listening on {interface}...")

    # Start the keypress listener in a separate thread
    keypress_thread = threading.Thread(target=stop_sniffing_on_keypress)
    keypress_thread.start()

    # Start sniffing the network traffic
    sniff(iface=interface, prn=process_packet, store=False)

    # Wait for the keypress thread to finish
    keypress_thread.join()
    print("Sniffing stopped.")

