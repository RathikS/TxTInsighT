from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import time

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Function for summarization
def summarize_text(text):
    # Tokenize the input text
    inputs = tokenizer("summarize: " + text, return_tensors="tf", max_length=5000, truncation=True)
    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = request.form["text"]
    summary = summarize_text(input_text)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
