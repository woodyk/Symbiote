#!/usr/bin/env python3
#
# deceptionapi.py

from flask import Flask, render_template_string, request, jsonify
import json
import symbiote.DeceptionDetection as dd

app = Flask(__name__)

# Initialize the DeceptionDetector
detector = dd.DeceptionDetector()

# HTML content for both index and result page with wider cards and proper scrolling
index_html = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deception Detector</title>
    <style>
        * {
            font-family: 'Courier new', sans-serif;
        }
        body {
            background-color: #000000;
            color: #ffffff;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            max-width: 800px;
            width: 90%;
            padding: 30px;
            background-color: #1e1e1e;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        h4 {
            text-align: center;
            color: #ff6f00;
        }
        label {
            font-size: 1.2em;
            color: #ffffff;
        }
        textarea {
            width: 90%;
            padding: 15px;
            margin-top: 10px;
            margin-bottom: 20px;
            background-color: #333;
            color: #ffffff;
            border: 1px solid #444;
            border-radius: 8px;
            font-size: 8;
            resize: vertical;
        }
        input[type="submit"] {
            width: 100%;
            padding: 15px;
            background-color: #ff6f00;
            color: #000000;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: inset 0 2px 0 #c05000, 0 2px 4px rgba(0, 0, 0, 0.4);
        }
        input[type="submit"]:hover {
            background-color: #e65a00;
        }
    </style>
</head>
<body>
    <div class="container">
        <h4>Deception Detection</h4>
        <form action="/analyze" method="post">
            <label for="input_text_or_url">Enter text or URL:</label><br>
            <textarea id="input_text_or_url" name="input_text_or_url" rows="10"></textarea><br>
            <input type="submit" value="Analyze">
        </form>
    </div>
</body>
</html>
'''

result_html = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deception Analysis Result</title>
    <style>
        * {
            font-family: 'Arial', sans-serif;
        }
        body {
            background-color: #000000;
            color: #ffffff;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            max-width: 800px;
            width: 90%;
            padding: 30px;
            background-color: #1e1e1e;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            overflow: auto;
            max-height: 80vh;
        }
        h4 {
            text-align: center;
            color: #ff6f00;
        }
        pre {
            background-color: #333;
            padding: 20px;
            border-radius: 8px;
            color: #ffffff;
            font-size: 1em;
            white-space: pre-wrap; /* Allows text to wrap */
            word-wrap: break-word; /* Ensures words break to fit */
            overflow-x: auto;
            max-height: 60vh; /* Ensure pre block doesn't overflow container */
            margin-bottom: 20px;
        }
        .pretty-json {
            font-family: monospace;
            white-space: pre;
            color: #ffb74d;
        }
        a {
            color: #ff6f00;
            text-decoration: none;
            display: block;
            margin-top: 20px;
            text-align: center;
            font-size: 1.1em;
        }
        a:hover {
            color: #e65a00;
        }
    </style>
    <script>
        // Function to format JSON with indentation and color coding
        function syntaxHighlight(json) {
            json = JSON.stringify(JSON.parse(json), undefined, 4);
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|\d+)/g, function (match) {
                var cls = 'number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'key';
                    } else {
                        cls = 'string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'boolean';
                } else if (/null/.test(match)) {
                    cls = 'null';
                }
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }

        window.onload = function() {
            var jsonElem = document.getElementById('json');
            jsonElem.innerHTML = syntaxHighlight(jsonElem.innerHTML);
        }
    </script>
</head>
<body>
    <div class="container">
        <h4>Deception Analysis Result</h4>
        <pre id="json" class="pretty-json">{{ result }}</pre>
        <a href="/">Analyze Another Text or URL</a>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/analyze', methods=['POST'])
def analyze():
    input_text_or_url = request.form['input_text_or_url']
    result = detector.analyze_text(input_text_or_url)
    return render_template_string(result_html, result=json.dumps(result, indent=4))

if __name__ == '__main__':
    app.run(debug=True)

