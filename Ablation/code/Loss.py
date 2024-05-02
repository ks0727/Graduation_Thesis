import numpy as np
from transformers import DonutProcessor

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

            