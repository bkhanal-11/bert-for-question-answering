import torch.nn as nn

class NextSentencePrediction(nn.Module):
    """
    2-class classification model to predict if the
    sentence is the next sentence or not
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(nn.Module):
    """
    Predicting original token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, d_model, vocab_size):
        """
        :param d_model: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, model, vocab_size):
        """
        :param vocab_size: total vocab size for masked_lm
        """
        super(BERTLM, self).__init__()
        self.bert = model
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, x_segment, mask=None):
        x = self.bert(x, x_segment, mask=mask)
        return self.next_sentence(x), self.mask_lm(x)
