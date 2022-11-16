# PICSTER
Abstract 
Since the dawn of the news revolution, the credibility of an article has always been questioned. The question of whether the article is good or not is always on our mind. We suggest a strategy that uses SOTA neural network designs to make it easier for you to determine whether an article is good or not. There are three sections to the methodology. Captioning for the image extracted from the article, text extraction from the article, and finally comparing the extracted text with the text obtained as caption with the help of a mathematical model. Our test data showed that the captioning architecture, CLIP, delivered outstanding results. We use OCR for text extraction of the article. It can partially work with blurry photographs. The text is summarized using BART. Then, to understand what the article is about, we use a Zero-Shot learning model that takes the extracted keywords as input and checks the probability of each term. The text collected using OCR is abstractedly summarised, and then both the keywords and the text are matched to provide a final result.
Keywords
Zero-Shot Learning 
Image Captioning 
OCR 
Articles 
Text Comparison
Introduction 
Articles are available on increasingly numerous platforms 24/7 and are also disseminated and consumed in digital networks via intermediaries such as social media platforms. People's news consumption habits are evolving. The first question that frequently comes to mind when we see an item in the newspaper or on the internet is whether it is worth a read or not. The quality of the content is in question. There are fantastic articles that everyone enjoys reading, and there are also poor pieces that no one wants to read. For humans, determining whether an article is good or not is a time-consuming endeavor because we must first read the entire piece before determining whether it was worth our time. Taking a glance over the article may not be sufficient, as there are many aspects to consider when determining the article's quality. People utilize heuristics and cues to evaluate information. Sometimes we are very attentive and thoroughly absorb information, but often we rely on mental shortcuts. Knowing whether the news is biased is essential given the volume of information available today and can strongly impact the public perception of the reported topics. A neutral article is always regarded favorably since it encourages you to form your own view, as opposed to biased pieces that aim to have you second-guess it. There are often strong and exaggerated emotions involved in an article that is biased which should not be the case as an article's main purpose is to provide unbiased information. Recognizing bias is also backbreaking work as you have to check every nook and cranny of the article be it the author, the tone, or the sources.
Applying the machine learning method can be of immense help in reducing the time consumed in the peer review procedure for research articles. Therefore, it makes sense to speed up this procedure to save everyone time.
Finding any bias in a newspaper piece is another issue. 
Another important aspect that we need to consider is the images used in the article which are a core part and can shape our view of the article. An image can have a totally different meaning from the content of the article which would be a sign of a bad article as the image should be something that works as a basis of the understanding of the article. There can be bias in the image too which should be considered.
All the machine learning models designed previously have focused on a single topic or a small set of fields and have been mainly used for classification purposes of the article's citations. Therefore, we propose in this paper to integrate modern nlp and computer vision to judge an article's reliability. Recent advancements in the field of deep learning help us compile all these different sections easily. 
The goal of this research is to scrutinize whether there is a possibility to evaluate the goodness of any article, be it a newspaper or an online article by the use of eclectic neural network architectures in conjunction. There should be a direct relationship between the image and the content of the article to infer its quality. Therefore the steps would be to extract the image present in the article if present and then extract all the content and finally develop a quality metric to define the standard. 
The specific research question would therefore be as follows: 
• How precisely can all these SOTA architectures judge the quality of the article based on the certain inputs provided to them. 
• Whether the methods employed are good enough to reach the goal of the article and if there are any better architectures to perform the following tasks. 
• Which of the features of the article (text, image, or both) are most useful to conclude the result.
Method
Determining the credibility of an article from an input document involves four steps. These include :
• Connecting Text and Images (CLIP) for image captioning
• Tesseract Optical Character Recognition(OCR) for text and image extraction
• Zero Shot Learning for comparative analysis
• BART for summarization

The overall architecture of picture is depicted in Figure 1.The article under question is here saved in pdf format.
The images and text of the article is extracted with the help of Tesseract Optical Character Recognition(OCR). Tesseract is free and open source. It is perfect for scanning clean documents and comes with pretty high accuracy and font variability since its training was comprehensive. It is the go-to tool for tasks such as scanning books, documents, and printed text on a clean white background. Tesseract 4.0 has neural net based OCR engine mode that supports high noise documents with improved accuracy. We used OpenCV, the standard library for computer vision and image processing, to de-noise and pre-process our documents as images. OpenCV is based on the BGR color format because back in time BGR color format was popular among camera manufacturers and software providers, which is why we first converted the image from RGB to BGR. We used pytesseract to communicate with the Tesseract OCR engine. The efficiency of tesseract depends on how well the data is cleaned to make the input simpler for images to extract. The extracted text is cleaned and saved.

We tried generating an appropriate caption for the extracted image with the help of CLIP. CLIP uses a contrastive learning approach to unify images and text and turn image classification into a text similarity problem. The model, pretrained on text and images pairs from the internet data, predicts the most relevant text snippet for a given image.


The corpus extracted from the document is summarised both abstractively and extractively. We used the transformers library from hugging face. BART, Bidirectional Autoregressive Transformers is a denoising autoencoder for pretraining sequence-to-sequence models. Numerous algorithms are implemented in this package. The first part of BART uses the bi-directional encoder of BERT to identify the best representation of its input sequence. The BERT encoder generates a vector that includes embedding vectors for each token in the input text sequence as well as sentence-level information. To transfer an input text sequence to the output, a decoder must read the tokens and sentence-level representation. As a result, BART excels at a variety of tasks like abstractive discourse, answering questions, and summarising. The saved text is fed into the Bart transformer model to generate an abstract highlighting the key ideas of the article, which we call the summary. 

The summary generated serves as the input for the keyword extraction with the use of python keyphrase extraction module. Each sentence from the summary of the document is selected as a keyphrase candidate. The candidates are ranked using a candidate
weighting function (unsupervised approaches). Lastly, the top-N highest weighted candidates, or the keyphrase with the highest confidence scores, are selected as keywords. These keywords are the labels. For example the labels generated from the summary are stored in a list as :
 
['india', 'battle', 'crucial', 'positive coronavirus cases', 'covid-19 outbreak']

These keywords are used as labels for the extracted image in order to perform zero shot analysis. We must encode both images and their describing labels. 
Zero-shot is a machine learning paradigm that introduces the idea of testing samples with class labels that were never observed during the initial training phase, the idea being more or less similar to how we human beings extrapolate our learnings to a new concept based on existing knowledge that we gather over time. The fact that the dataset labels are in text format, much like the input, gives us an edge in the field of NLP. This gives language models a good place to start when used as zero-shot learners because they can comprehend the meaning of labels because they have some level of natural language understanding.  This is a classification task framed into an NLI problem. Zero-shot classification is a technique which allows us to associate an appropriate label with a piece of text. The relatedness of the labels with the summary is calculated and the summary is most likely to be an excerpt of the label with the highest accuracy score.
                         
If the summary generated was :
PM Modi is closely monitoring the situation daily and discussing ways to further strengthen India's preparedness. The next fifteen days are crucial in India's battle against the COVID-19 outbreak. All positive coronavirus cases in India so far have been related to people coming from abroad. 
And is associated with the labels generated above ['india', 'battle', 'crucial', 'positive coronavirus cases', 'covid-19 outbreak']
The result would be :

![image](https://user-images.githubusercontent.com/65299277/202278624-d434e657-6599-4ea0-95f8-2d00c3af820b.png)

Comparative analysis between the image caption and the category (label with higest probability)

Related Work 
All of the architectures listed in this paper have each been extensively used, but not all of them in combination. Together, these SOTA systems can function flawlessly and be used in various situations. There has previously been research on the evaluation of published articles, but not of pieces from newspapers or websites.
The main idea behind the methodology was to combine different architectures and check how well these different architectures work together when combined. They each serve a unique purpose, they all work together to help us complete our results.

Results and Discussion 
In this way, the crux of the article is quickly generated which allows the readers to swiftly skim through the highlights of a large quantity of textual content while saving enough time to examine the quality of information. The image caption, extractive, and abstractive summaries are compared using zero shot learning.
