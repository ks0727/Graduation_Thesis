import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch.nn as nn
from typing import Any,Tuple
from Loss import BERT_COS_SIM
from Cross_Attention_Map import CrossAttentionMap
import os

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.encoder.config.output_scores = True
model.config.output_attentions = True

def remove_dence_layer(layer_block_pairs:Tuple[int,int])->None:
    for layer,block in layer_block_pairs:
        model.encoder.encoder.layers[layer].blocks[block].attention.output.dense = nn.Identity()

layer_block_pairs_to_remove = [(3,0),(3,1)] #substitute (layer,block) pairs into this list
remove_dence_layer(layer_block_pairs_to_remove)

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

path = './result/CrossAttentionMaps/Dence_Layer_Ablation'
cross_attns = outputs.cross_attentions

cross_attn_map = CrossAttentionMap(cross_attns=cross_attns,path=path)
#cross_attn_map.get_cross_attn_maps()
#seq = get_ids_from_tokens(outputs.scores)
#res = processor.tokenizer.batch_decode(seq)

decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print("-------------------------------------------")
print("output : ", processor.token2json(sequence))
print("-------------------------------------------")


prediction = softmax(outputs.scores)
criterion = BERT_COS_SIM(query=ans_label,sentence=sequence)
loss = criterion.forward()
print(f'loss : {loss[0][0]}')


dence_1_sims = [0.85889,0.98868,0.99364,0.99883,0.99208,0.99208,0.99208,0.99583,0.99208,0.99208,0.99208,0.99208,0.98941,0.99208,0.99606,0.99208,0.99208,0.99208,0.99208,0.99208]
dence_1_x = list(range(len(dence_1_sims)))
dence_1_labels = ["(1,1)","(1,2)","(2,1)","(2,2)","(3,1)","(3,2)","(3,3)","(3,4)","(3,5)","(3,6)","(3,7)","(3,8)","(3,9)","(3,10)","(3,11)","(3,12)","(3,13)","(3,14)","(4,1)","(4,2)"]
fig1 = plt.figure(figsize=(12,10))
ax = fig1.add_subplot(1,1,1)
ax.bar(dence_1_x,dence_1_sims,tick_label=dence_1_labels)
ax.set_xlabel("(layer,block) pairs removed")
ax.set_ylabel("cosine similarity")
ax.set_title("cosine similarities with removing one dence layer")
fig1.savefig(os.path.join(path,'dence_1_ablation'))

dence_2_sims = [0.98243,0.98875,0.99208,0.99677,0.99208,0.99298,0.98941,0.98434,0.98460,0.992087]
dence_2_x = list(range(len(dence_2_sims)))
dence_2_labels = ["(1,1),(1,2)","(2,1),(2,2)","(3,1),(3,2)","(3,3),(3,4)","(3,5),(3,6)","(3,7),(3,8)","(3,9),(3,10)","(3,11),(3,12)","(3,13),(3,14)","(4,1),(4,2)",]
fig2 = plt.figure(figsize=(12,10))
ax = fig2.add_subplot(1,1,1)
ax.bar(dence_2_x,dence_2_sims,tick_label=dence_2_labels)
ax.set_xlabel("(layer,block) pairs removed")
ax.set_ylabel("cosine similarity")
ax.set_title("cosine similarities with removing two dence layers")
fig2.savefig(os.path.join(path,'dence_2_ablation'))


dence_3_sims = [0.566753,0.565852,0.99672,0.99421,0.99542,0.96515,0.565852]
dence_3_x = list(range(len(dence_3_sims)))
dence_3_labels = ["(1,1),(1,2),(2,1)","(2,2),(3,1),(3,2)","(3,3),(3,4),(3,5)","(3,6),(3,7),(3,8)","(3,9),(3,10),(3,11)","(3,12),(3,13),(3,14)","(3,14),(4,1),(4,2)"]
fig3 = plt.figure(figsize=(12,10))
ax = fig3.add_subplot(1,1,1)
ax.bar(dence_3_x,dence_3_sims,tick_label=dence_3_labels)
ax.set_xlabel("(layer,block) pairs removed")
ax.set_ylabel("cosine similarity")
ax.set_title("cosine similarities with removing three dence layers")
fig3.savefig(os.path.join(path,'dence3_ablation'))



dence_4_sims = [0.56039,0.94285,0.96676,0.96333,0.95679]
dence_4_x = list(range(len(dence_4_sims)))
dence_4_labels = ["(1,1),(1,2),(2,1),(2,2)","(3,1),(3,2),(3,3),(3,4)","(3,5),(3,6),(3,7),(3,8)","(3,9),(3,10),(3,11),(3,12)","(3,13),(3,14),(4,1),(4,2)",]
fig4 = plt.figure(figsize=(12,10))
ax = fig4.add_subplot(1,1,1)
ax.bar(dence_4_x,dence_4_sims,tick_label=dence_4_labels)
ax.set_xlabel("(layer,block) pairs removed")
ax.set_ylabel("cosine similarity")
ax.set_title("cosine similarities with removing four dence layers")
fig4.savefig(os.path.join(path,'dence4_ablation'))

dence_5_sims = [0.56039,0.91759,0.929557,0.70510]
dence_5_x = list(range(len(dence_5_sims)))
dence_5_labels = ["(1,1),(1,2),(2,1),(2,2),(3,1)","(3,2),(3,3),(3,4),(3,5),(3,6)","(3,7),(3,8),(3,9),(3,10),(3,11)","(3,12),(3,13),(3,14),(4,1),(4,2)"]
fig5 = plt.figure(figsize=(12,10))
ax = fig5.add_subplot(1,1,1)
ax.bar(dence_5_x,dence_5_sims,tick_label=dence_5_labels)
ax.set_xlabel("(layer,block) pairs removed")
ax.set_ylim([0,1])
ax.set_ylabel("cosine similarity")
ax.set_title("cosine similarities with removing five dence layers")
fig5.savefig(os.path.join(path,'dence5_ablation'))
