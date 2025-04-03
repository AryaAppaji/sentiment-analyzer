import gradio as gr
from transformers import pipeline

def analyze_sentiment(text: str):
    model_path = r"ai_models\models--distilbert--distilbert-base-uncased-finetuned-sst-2-english\snapshots\714eb0fa89d2f80546fda750413ed43d93601a13"
    sentiment = pipeline(
        task="sentiment-analysis",
        model= model_path
    )

    result = sentiment(text)
    return result[0]["label"], result[0]["score"]

# print(analyze_sentiment("It's a beautiful day"))

interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=[
        gr.Textbox(label="Please enter your text", lines=6),
    ],
    outputs=[
        gr.Textbox(label="Output"),
        gr.Textbox(label="Score"),
    ],
    title="Sentiment Analyzer",
    description="This is an AI app which analyzes the sentiment from the given text based on the <a href='https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english'>distilbert/distilbert-base-uncased-finetuned-sst-2-english</a> model from <a href='https://huggingface.co/'>huggingface.co</a>",
    article="Created by Arya Appaji, March 2025",
    submit_btn="Analyze Sentiment"
)

interface.launch()
