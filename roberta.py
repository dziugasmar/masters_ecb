"""
RoBERTa Sentiment Classifier for Excel Speech Data

Usage:
    python roberta.py --input path/to/input.xlsx

- The script will process the Excel file, classify each speech row, and save results as 'input_with_sentiment.xlsx' and '.csv'.
- Requires HuggingFace transformers, torch, pandas, and openpyxl.
"""
import argparse
import os
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import pandas as pd
import re

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def process_file(input_path, hf_token=None):
    # Load the data
    df = pd.read_excel(input_path)
    if 'speech' not in df.columns:
        raise ValueError("Input file must have a 'speech' column.")
    df['cleaned_speech'] = df['speech'].apply(clean_text)

    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa", do_lower_case=True, do_basic_tokenize=True)
    model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/FOMC-RoBERTa", num_labels=3)
    config = AutoConfig.from_pretrained("gtfintechlab/FOMC-RoBERTa")
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, config=config, device=0 if torch.cuda.is_available() else -1, framework="pt")

    # Process each row
    sentiments = []
    confidences = []
    for speech_text in df['cleaned_speech']:
        if not speech_text or len(speech_text) < 10:
            sentiments.append("N/A")
            confidences.append(0.0)
            continue
        try:
            if len(speech_text) > 25000:
                speech_text = speech_text[:25000]
            result = classifier(speech_text, truncation=True)
            sentiments.append(result[0]['label'])
            confidences.append(result[0]['score'])
        except:
            sentiments.append("ERROR")
            confidences.append(0.0)
    df['sentiment'] = sentiments
    df['confidence'] = confidences

    # Save results
    base, ext = os.path.splitext(input_path)
    out_xlsx = f"{base}_with_sentiment.xlsx"
    out_csv = f"{base}_with_sentiment.csv"
    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_xlsx} and {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Classify sentiment of speeches in an Excel file using RoBERTa.")
    parser.add_argument('--input', '-i', required=True, help='Path to input Excel file')
    parser.add_argument('--hf_token', default=None, help='HuggingFace token (optional, can also use env HUGGINGFACE_TOKEN)')
    args = parser.parse_args()
    hf_token = args.hf_token or os.environ.get('HUGGINGFACE_TOKEN', None)
    process_file(args.input, hf_token)

if __name__ == "__main__":
    main()



