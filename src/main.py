from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Real-Time Speech Recognizer</h1>
    <p>Server inside Docker!</p>
    <ul>
        <li>Google Speech-to-Text: ready</li>
        <li>Speaker ID (resemblyzer): ready</li>
        <li>Flask + SocketIO: ready</li>
    </ul>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
