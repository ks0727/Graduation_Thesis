import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch.nn as nn
from typing import Any
from Loss import BERT_COS_SIM
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
#model.encoder.config.output_scores = True
model.encoder.encoder.layers[0].blocks[0].layernorm_before = nn.Identity()
model.encoder.encoder.layers[0].blocks[0].layernorm_after = nn.Identity()
model.encoder.encoder.layers[0].blocks[1].layernorm_before = nn.Identity()
model.encoder.encoder.layers[0].blocks[1].layernorm_after = nn.Identity()
model.encoder.encoder.layers[1].blocks[0].layernorm_before = nn.Identity()
model.encoder.encoder.layers[1].blocks[0].layernorm_after = nn.Identity()
model.encoder.encoder.layers[1].blocks[1].layernorm_before = nn.Identity()
model.encoder.encoder.layers[1].blocks[1].layernorm_after = nn.Identity()
model.encoder.encoder.layers[2].blocks[0].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[0].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[1].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[1].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[2].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[2].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[3].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[3].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[4].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[4].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[5].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[5].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[6].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[6].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[7].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[7].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[8].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[8].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[9].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[9].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[10].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[10].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[11].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[11].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[12].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[12].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[2].blocks[13].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[2].blocks[13].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[3].blocks[0].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[3].blocks[0].layernorm_after = nn.Identity()
#model.encoder.encoder.layers[3].blocks[1].layernorm_before = nn.Identity()
#model.encoder.encoder.layers[3].blocks[1].layernorm_after = nn.Identity()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
#print(model.encoder)
dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
task_prompt = "<s_iitcdip>"

decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

pixel_values = processor(image, return_tensors="pt").pixel_values

ans_label = "11:14 to 11:39 a.m. Coffee Break Coffee will be served for men and women in the lobby adjacent to exhibit area. Please move into exhibit area. "\
"(Exhibits Open) 11:39 a.m. TRRF GENERAL SESSION (PART I). Presiding: Lee A. Waller TRRF Vice President 11:39 to 11:44 a.m. “Introductory Remarks” Lee A. Waller, "\
"TRRF Vice Presi- dent 11:44 a.m. to 12:25 p.m. Individual Interviews with TRRF Public Board Members and Sci- entific Advisory Council Mem- bers Conducted by TRRF"\
"Treasurer Philip G. Kuehn to get answers which the public refrigerated warehousing industry is looking for. Plus questions from the floor. Dr. Emil M. Mrak,"\
"University of Cal- ifornia, Chairman, TRRF Board; Sam R. Cecil, University of Georgia College of Agriculture; Dr. Stanley Charm,"\
"Tufts University School of Medicine; Dr. Robert H. Cotton, ITT Continental Baking Company; Dr. Owen Fennema, "\
"University of Wis- consin; Dr. Robert E. Hardenburg, USDA. 12:25 to 12:58 p.m. Questions and Answers 12:58 to 4:00 p.m. "\
"Exhibits Open Capt. Jack Stoney Room 2:00 to 5:00 p.m. TRRF Scientific Advisory Council Meeting Ballroom Foyer"
label_ids = processor.tokenizer(ans_label,add_special_tokens = False,return_tensors="pt").input_ids[0]
label_ids = label_ids.to('cpu').detach().numpy().copy()
outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    output_scores=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

#By using this function, you can get the sequence of token ids whose probability is the highest among all vocabrary
def get_ids_from_tokens(scores:torch.Tensor):
    seq = []
    for i in range(len(scores)):
        sci = scores[i]
        sci_np = sci.to('cpu').detach().numpy().copy()
        seq.append(np.argmax(sci_np))
    return seq

def cross_entropy_loss(y_true,y_pred):
    loss = 0
    for i in range(len(y_pred)):
        print(label_ids[i])
        loss += (-1*np.log(y_pred[i][label_ids[i]]))
    return loss

def softmax(scores:torch.Tensor):
    h = len(scores) #the length of the sequence
    w = scores[0].size()[1] # the length of the embeded dimension
    y_pred = np.zeros((h,w))
    for i in range(len(scores)):
        sci = scores[i]
        sci_np = sci.to('cpu').detach().numpy().copy()
        e_x = np.exp(sci_np-np.max(sci_np))
        y_pred[i] = e_x/ e_x.sum()
    return y_pred

prediction = softmax(outputs.scores)




#seq = get_ids_from_tokens(outputs.scores)
#res = processor.tokenizer.batch_decode(seq)

decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print("-------------------------------------------")
print("output : ", processor.token2json(sequence))
print("-------------------------------------------")

criterion = BERT_COS_SIM(query=ans_label,sentence=sequence)
loss = criterion.forward()
print(f'similarity between two sentences : {loss[0][0]}')
