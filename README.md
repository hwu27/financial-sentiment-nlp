# financial-sentiment-nlp

This project details how to use a transformer model (BERT) with **Pytorch** and **Selenium** to both retrieve financial news and to create a sentiment score for a specific company/stock.

Installation (Linux):

1. pip install selenium (don't worry about ChromeDriver, it was added in previous patches)
2. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (Note: I am using Cuda.11.8)
3. pip install transformers
4. pip install -U scikit-learn
5. pip install pandas
6. pip install numpy

For accelerated GPU training (Recommended):

-----

7. Install cuda: https://docs.nvidia.com/cuda/wsl-user-guide/index.html 
8. Install cuDNN: https://developer.nvidia.com/rdp/cudnn-archive

-----

So how does it work?
1. I first trained the transformer model BERT through the use of the transformers model from HuggingFace using the Sentiment Analysis for Financial News on Kaggle

![image](https://github.com/hwu27/financial-sentiment-nlp/assets/130116077/ae184e71-5fcc-47f8-bac1-9c3c4f9f6af0)


2. I then tested the model on the validation dataset that we obtained from splitting the dataset using the CustomDataset class
   
![image](https://github.com/hwu27/financial-sentiment-nlp/assets/130116077/401fa95e-b30d-4417-8a4c-74cd3880ab13)

3. Next, I created a Selenium Webdriver to search a financial news outlet (in this case, CNBC)

![image](https://github.com/hwu27/financial-sentiment-nlp/assets/130116077/1f544edc-a083-4500-baad-b977e34bf716)

Selenium will extract key information from the website, which then we pass back into the model for sentiment analysis.
Note, I only extract the title in this project, but the Selenium script is made in a way so that you can extract more from the website if wanted. I just did it for the sake of time.

![image](https://github.com/hwu27/financial-sentiment-nlp/assets/130116077/1b8b1287-766f-4013-b893-1028dde301f0)


![image](https://github.com/hwu27/financial-sentiment-nlp/assets/130116077/2bd7f85e-3211-4df7-a97d-0c8ec2300ebb)

To improve the model, add more data and extra more data from different websites.

Acknowledgements:

Dataset: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

Model Inspiration: https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb (Amazing tutorial and main driver for this project)

