import torch
import numpy as np
from transformers import DonutProcessor,BertTokenizer,BertModel

class CELWithDiffenretLength():
    def __init__(self,seq_pred,seq_truth):
        self.seq_pred = seq_pred
        self.seq_truth = seq_truth
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        # make the ndarray for prediction
        self.truth_labels = processor.tokenizer(seq_truth,add_special_tokens=False,return_tensors="pt").input_ids[0]
        self.truth_labels = self.truth_labels.to('cpu').detach().numpy().copy()

    def forward(self):
        iters = min(self.truth_labels.shape[0],self.seq_pred.shape[0])
        loss = 0
        for i in range(iters):
            loss += - np.log(self.seq_pred[i][self.truth_labels[i]])
        loss /= iters
        return loss
    
class BERT_COS_SIM():
    def __init__(self,query,sentence) -> None:
        self.query = query
        self.sentence = sentence
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenized_sentence = self.tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")
        self.tokenized_query = self.tokenizer(query,padding=True,truncation=True,return_tensors="pt")
        
    def forward(self):
        with torch.no_grad():
            sentence_output = self.model(**self.tokenized_sentence)
            query_output = self.model(**self.tokenized_query)
        
        query_embedding = query_output[0][:,0,:].numpy()
        sentence_embedding = sentence_output[0][:,0,:].numpy()
        cos_sim = np.inner(query_embedding,sentence_embedding)
        return cos_sim
