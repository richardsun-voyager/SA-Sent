# Learning Latent Opinions for Aspect-level Sentiment Classification 

This code is a modified version based on Bailin's [origin repository](https://github.com/berlino/SA-Sent). Contributions:

- The code was transferred from python 2.7 to 3.6
- Word embeddings such as Glove have been replaced by Elmo 
- Train and test in batches
- Add classical attention model to make comparisons

## Requirements

* python 3.6
* pytorch, spacy, numpy, bs4, Elmo
* glove for pre-trained vectors

## Steps

Note, in order to implement the following steps correctly, please enter the root folder "SA-Sent" in your terminal.

1. Download data from SemEval 2014 task 4, actually for the sake of convenience, we have included the dataset in the folder "data/2014".

2. Install spacy and language package with commands:

   ```
   pip install spacy
   python -m spacy download en
   ```

   ​

3. Install Elmo environment,  [anaconda](https://www.anaconda.com/download/#macos) packages are prerequisites:

  ```
  #Create a virtual environment named allennlp
  conda create -n allennlp python=3.6
  #Activate the virtual environment named allennlp 
  source activate allennlp
  #Install allennlp and its dependent packages
  pip install allennlp
  ```

  ​

4. Data has been processed in advance and put in the folder "data". If necessray, you can preprocess the original data by running command in your terminal, make sure you have activated "allennlp" environment:

  ```
  #Tokenize sentences and serialize them in local folder
  python reader.py
  ```

  ​

5. Download pretrained Elmo [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)  and [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json), put them in folder 'data/elmo'.

6. Run the training and testing code, the results will echo in the terminal as well as in the log. If you have GPU resource, you can turn on GPU by setting "if_gpu=True" in the file "config.py":

   ```
   python train.py
   ```

   Note if you code runs in a remote server,  and you do not expect any interruptions like disconnection, you can type such command instead in your terminal:

   ```
   screen python train.py
   ```

7. All the configuration hypeparameters are listed in the file "config.py", you can change the folder paths, epoch numbers accordingly.

8. You can run a classical attention model using such commands in the terminal:

   ```
   python train_att.py
   ```

   ​


## Question?

Feel free to email me xiaobing_sun@sutd.edu.sg.
