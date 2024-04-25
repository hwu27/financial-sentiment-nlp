# financial-sentiment-nlp

This project details how to use a transformer model (BERT) with Pytorch and Selenium to both retrieve financial news and to create a sentiment score for a specific company/stock.

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
