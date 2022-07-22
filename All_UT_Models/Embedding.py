import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec 
import torch

class Embeddings():
  """
  Class to create Word Embeddings from data
  """
  def __init__(self, sentences, size=300, window=3, epochs=75, min_count=2, embedding_type="fast_text", skip_gram=1):
    """
    Arguments:
          sentences: data to create embeddings from
          size: embedding size
          window: window to select data
          epochs: number of epochs to train the model
          min_count: min count for words to include
          embedding_type: model for creating embedding "w2v" for word2vec model and "fast_text" for FastText model
          skip_gram: 0 or 1 either train model using skipgram or CBOW
    """
    self.SIZE = size
    self.WINDOW = window
    self.EPOCH = epochs
    self.MIN_COUNT = min_count
    self.TYPE = embedding_type
    self.SKIP_GRAM = skip_gram
    self.DOCUMENTS = [_text.split(" ") for _text in sentences]       

  def Model(self):
    """
    Method to Instantiate the model
    """
    if self.TYPE == "w2v":
        self.MODEL = gensim.models.word2vec.Word2Vec(size=self.SIZE, window=self.WINDOW, min_count=self.MIN_COUNT, workers=8, sg=self.SKIP_GRAM)  
    elif self.TYPE == "fast_text":
        self.MODEL = gensim.models.fasttext.FastText(size=self.SIZE, window=self.WINDOW, min_count=self.MIN_COUNT, workers=8, sg=self.SKIP_GRAM)
    else:
        raise Exception('Embedding Model not valid')

  def BuildVocab(self):
    """
    Method to build the vocabulary for the model
    """
    self.MODEL.build_vocab(self.DOCUMENTS)

  def VocabSize(self):
    """
    Method to get size of the vocabulary
    """
    words = len(self.MODEL.wv.vocab.keys())
    return words

  def Train(self):
    """
    Method to train the model
    """
    self.MODEL.train(self.DOCUMENTS, total_examples=len(self.MODEL.wv.vocab.keys()), epochs=self.EPOCH)

  def GetModel(self, load=False, path="./"):
    """
    Method to get trained model
    Parameters:
          load: True if want to load existing model
          path: path of the model
    """
    if load: self.MODEL = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
      self.Model()
      self.BuildVocab()
      self.Train()

  def LoadPretrainedModel(self, path='./model.bin', binary=True):
    """
    Method to load pre-trained model
    Parameters:
          path: path to load model from
          binary: whether binary format or not 
    """
    if binary: self.pre_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    else: self.pre_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
 
  def GetUpdatedModel(self, path='./model.bin'):
    """
    Method to update the already existing models vocabulary with new vocabulary and train the model
    Parameters:
          path: path to load model from
    """
    self.Model()
    self.BuildVocab()
    self.MODEL.build_vocab([list(self.pre_model.vocab.keys())], update=True)
    if self.TYPE=="w2v": self.MODEL.intersect_word2vec_format(path, binary=True)
    self.Train()

  def GetEmbeddingMatrix(self):
    """
    Method to get embedding matrix
    """
    return self.MODEL.wv.vectors

  def SaveModel(self, path="./", filename='model.bin'):
    """
    Method to save the model
    Parameters:
          path: path to save model to
          filename: filename and format to save file in
    """    
    self.MODEL.wv.save_word2vec_format(path+filename, binary=True)