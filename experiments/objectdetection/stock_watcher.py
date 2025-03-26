#!/usr/bin/env python3
#
# stock_watcher.py
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PIL import Image
import requests
from io import BytesIO
import os

def load_image(image_input):
    """Load an image from a file path or URL."""
    if image_input.startswith('http://') or image_input.startswith('https://'):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    elif os.path.isfile(image_input):
        return cv2.imread(image_input)
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")

def preprocess_image(image):
    """Convert image to grayscale and apply edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def extract_axis_labels(image):
    """Extract the x and y axis labels using OCR."""
    # Use pytesseract to extract text
    x_axis_labels = []
    y_axis_labels = []

    # Convert image to grayscale and invert (assuming dark text on light background)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray)

    # OCR on inverted image to extract text
    text_data = pytesseract.image_to_data(inverted_image, output_type=pytesseract.Output.DICT)

    for i in range(len(text_data['text'])):
        x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
        text = text_data['text'][i]

        # Heuristic: y-axis labels are usually on the left side, x-axis at the bottom
        if x < image.shape[1] * 0.2:  # Near the left side
            y_axis_labels.append((y, text))
        elif y > image.shape[0] * 0.8:  # Near the bottom
            x_axis_labels.append((x, text))

    return sorted(x_axis_labels), sorted(y_axis_labels, key=lambda x: x[0])

def detect_candlesticks(image, edges):
    """Detect candlesticks and return their positions and sizes."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candlesticks = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Identify candlestick shapes (rectangles with appropriate aspect ratio)
        if 10 < w < 40 and 20 < h < 100 and aspect_ratio < 1:
            candlesticks.append((x, y, w, h))

    return candlesticks

def extract_candlestick_data(candlesticks, image, y_axis_labels):
    """Extract OHLC data from candlesticks."""
    ohlc_data = []
    y_axis_positions = [label[0] for label in y_axis_labels]
    y_axis_values = [float(label[1].replace(',', '')) for label in y_axis_labels]

    def map_y_to_price(y):
        return np.interp(y, y_axis_positions, y_axis_values)

    for x, y, w, h in candlesticks:
        is_bullish = np.mean(image[y:y+h, x:x+w, 1]) > np.mean(image[y:y+h, x:x+w, 2])  # Green vs. Red
        open_price = y + h if is_bullish else y
        close_price = y if is_bullish else y + h
        high_price = y
        low_price = y + h

        ohlc_data.append({
            'x': x,
            'open': map_y_to_price(open_price),
            'high': map_y_to_price(high_price),
            'low': map_y_to_price(low_price),
            'close': map_y_to_price(close_price),
            'is_bullish': is_bullish
        })

    return ohlc_data

def plot_extracted_data(ohlc_data, x_axis_labels):
    """Plot the extracted OHLC data using matplotlib."""
    fig, ax = plt.subplots()

    for data in ohlc_data:
        color = 'green' if data['is_bullish'] else 'red'
        ax.plot([data['x'], data['x']], [data['low'], data['high']], color=color)
        ax.plot([data['x'] - 0.2, data['x'] + 0.2], [data['open'], data['open']], color=color)
        ax.plot([data['x'] - 0.2, data['x'] + 0.2], [data['close'], data['close']], color=color)

    x_ticks = [label[0] for label in x_axis_labels]
    x_labels = [label[1] for label in x_axis_labels]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)

    plt.title('Extracted Candlestick Data')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

def predict_next_candlestick(ohlc_data):
    """Predict the next candlestick based on linear regression."""
    X = np.array([data['x'] for data in ohlc_data]).reshape(-1, 1)
    y = np.array([data['close'] for data in ohlc_data])

    model = LinearRegression()
    model.fit(X, y)

    next_x = np.array([[X[-1][0] + (X[1][0] - X[0][0])]])
    next_close = model.predict(next_x)

    predicted_candlestick = {
        'x': next_x[0][0],
        'open': y[-1],
        'close': next_close[0],
        'high': max(y[-1], next_close[0]),
        'low': min(y[-1], next_close[0]),
        'is_bullish': next_close[0] > y[-1]
    }

    ohlc_data.append(predicted_candlestick)
    return ohlc_data

def analyze_image(image_input):
    """Main function to analyze the image and predict the next candlestick."""
    image = load_image(image_input)
    edges = preprocess_image(image)

    x_axis_labels, y_axis_labels = extract_axis_labels(image)
    candlesticks = detect_candlesticks(image, edges)

    if not x_axis_labels or not y_axis_labels:
        print("Unable to detect axes labels in the image.")
        return

    ohlc_data = extract_candlestick_data(candlesticks, image, y_axis_labels)
    plot_extracted_data(ohlc_data, x_axis_labels)

    ohlc_data_with_prediction = predict_next_candlestick(ohlc_data)
    plot_extracted_data(ohlc_data_with_prediction, x_axis_labels)

# Example usage with a local file path or a URL:
image_input = "https://www.amcharts.com/wp-content/uploads/2019/10/demo_14592_none-11.png"  # Replace with the actual path or URL
analyze_image(image_input)

