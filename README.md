# PICSTER : Image Extraction and Text Summarization Pipeline

This project focuses on evaluating the quality and reliability of articles using a combination of state-of-the-art (SOTA) machine learning techniques, including Zero-Shot Learning, Image Captioning, and Optical Character Recognition (OCR). The pipeline involves extracting text and images from articles, generating summaries, and comparing these summaries with image captions to identify bias and relevance. By leveraging CLIP for image-text alignment, BART for text summarization, and Zero-Shot Learning for classification, the system provides a comprehensive analysis of an article's content. The ultimate goal is to develop a reliable metric for assessing the quality of articles, helping readers quickly determine their worthiness without reading the entire text.

## Prerequisites
Before running the pipeline, ensure you have the following software and dependencies installed:

- Python 3.x
- Google Colab or a local environment with GPU support (recommended for faster processing)
- Required Python libraries (as listed below)

## Setup Instructions
1. Clone the Repository :
   - Clone the image-extraction repository to your working directory :
   ```bash
   !git clone https://github.com/harvardnlp/image-extraction/
   cd /content/image-extraction
   ```
3. Install Dependencies :
   - Update the system and install necessary packages:
   ```bash
   !apt-get update
   !bash setup.sh
   ```
4. Install PyTorch:
   - Install the specific version of PyTorch compatible with CUDA:
   ```bash
   !pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```

5. Download PDF Files:
   - Download sample PDF files into the pdfs directory:
   ```bash
   !mkdir pdfs
   cd /content/image-extraction/pdfs
   !wget https://openreview.net/pdf?id=mllQ3QNNr9d -O mllQ3QNNr9d.pdf
   !wget https://openreview.net/pdf?id=q2noHUqMkK -O q2noHUqMkK.pdf
   ```

6. Extract Images from PDFs:
   - Run the image extraction script to extract images from the downloaded PDFs:
   ```bash
   cd /content/image-extraction
   !mkdir pics
   !python run.py /content/image-extraction/pdfs /content/image-extraction/pics
   ```

7. OCR and Text Processing:
   - Install Tesseract OCR and Pytesseract:
   ```bash
   !sudo apt install tesseract-ocr
   !pip install pytesseract
   ```
   - Process the extracted images and extract text using Tesseract:
   ```bash
   import pytesseract
   import cv2
   image = cv2.imread('/content/image-extraction/dataset/q2noHUqMkK.png')
   img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   ext_text = pytesseract.image_to_string(img_rgb)
   print(ext_text)
   ```
     

8. Text Cleaning:
   - Clean and process the extracted text by removing emails, URLs, dates, and converting numbers to words:
   ```bash
   clean_text = " ".join(ext_text.split())
   # Further cleaning steps as provided in the script
   ```
   
9. Summarization and Keyphrase Extraction:
   - Install the Huggingface Transformers library:
   ```bash
   !git clone https://github.com/huggingface/transformers && cd transformers
   !pip install -q ./transformers
   ```
   - Use BART model to summarize the cleaned text:
   ```bash
   from transformers import pipeline
   summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
   summary = summarizer(clean_text, max_length=130, min_length=30, do_sample=False)
   print(summary)
   ```
   - Extract keyphrases using pke:
   ```bash
   import pke
   extractor = pke.unsupervised.TopicRank()
   extractor.load_document(input=summary, language='en')
   extractor.candidate_selection()
   extractor.candidate_weighting()
   keyphrases = extractor.get_n_best(n=5, stemming=False)
   print(keyphrases)
   ```
     
10. Zero-shot Classification:
   - Use a zero-shot classification model to classify the summarized text based on extracted keyphrases:
   ```bash
   from transformers import pipeline
   zero_shot_classifier = pipeline("zero-shot-classification")
   result = zero_shot_classifier(sequences=summary, candidate_labels=key_words, multi_class=True)
   print(result["labels"], result["scores"])
   ```
