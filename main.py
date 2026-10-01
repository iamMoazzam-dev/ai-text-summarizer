# from transformers import pipeline

# summarizer = pipeline("summarization")

# text = """
# Artificial intelligence is changing the way people work and learn.
# Generative AI can create text, summarize documents, answer questions,
# write code, and help users automate many tasks.
# """

# result = summarizer(text)

# print(result)

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model name
model_name = "sshleifer/distilbart-cnn-12-6"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Text to summarize
# text = """
# Artificial intelligence is changing the way people work and learn.
# Generative AI can create text, summarize documents, answer questions,
# write code, and help users automate many tasks.
# """
# text = input("Enter text to summarize: ")

# Convert text into tokens

def summarize_text(text):

    if not text.strip():
        return "Please enter some text to summarize."

    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

# Generate summary
    outputs = model.generate(
        **inputs,
        min_length=10,
        max_new_tokens=50,
        do_sample=False
    )

# Convert tokens back to text
    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary

# gradio input 
demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=12,
        label="Input Text",
        placeholder="Enter your text here..."
    ),
    outputs=gr.Textbox(
        lines=7,
        label="Generated Summary"
    ),
    
    title="AI Text Summarizer",
    description="Enter a paragraph and generate an AI-powered summary.",
    submit_btn="Summarize",
    clear_btn="Clear"
)


demo.launch()