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
model.config.output_attentions = True


def remove_layer_norm_before(layer_block_pairs:Tuple[int,int])->None:
    for layer,block in layer_block_pairs:
        model.encoder.encoder.layers[layer].blocks[block].layernorm_before = nn.Identity()
def remove_layer_norm_after(layer_block_pairs:Tuple[int,int])->None:
    for layer,block in layer_block_pairs:
        model.encoder.encoder.layers[layer].blocks[block].layernorm_after = nn.Identity()

layer_block_pairs_to_remove = [(3,1)] #substitute (layer,block) pairs into this list
remove_layer_norm_before(layer_block_pairs_to_remove)
remove_layer_norm_after(layer_block_pairs_to_remove)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
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
print(label_ids)
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



decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
print(len(decoded_results))
exit()
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print("-------------------------------------------")
print("output : ", processor.token2json(sequence))
print("-------------------------------------------")


path = './result/CrossAttentionMaps/LN_Both_Ablation'
cross_attns = outputs.cross_attentions

cross_attn_map = CrossAttentionMap(cross_attns=cross_attns,path=path)
cross_attn_map.get_cross_attn_maps(sentence=sequence)
prediction = softmax(outputs.scores)


criterion = BERT_COS_SIM(query=ans_label,sentence=sequence)
loss = criterion.forward()
print(f'similarity between two sentences : {loss[0][0]}')

# ln_before_sims = [0.93750,0.97338,0.99057,0.644617,0.570387,0.534322,0.62194,0.605492,0.534322,0.480766,0.5668034,0.582249,0.60666,0.495132,0.683659,0.683659,0.683659,0.499866,0.686455,0.686455]
# ln_before_x = list(range(len(ln_before_sims)))
ln_labels = ["(1,1)","(1,2)","(2,1)","(2,2)","(3,1)","(3,2)","(3,3)","(3,4)","(3,5)","(3,6)","(3,7)","(3,8)","(3,9)","(3,10)","(3,11)","(3,12)","(3,13)","(3,14)","(4,1)","(4,2)"]
# fig1 = plt.figure(figsize=(12,10))
# ax = fig1.add_subplot(1,1,1)
# ax.bar(ln_before_x,ln_before_sims,tick_label=ln_labels)
# ax.set_xlabel("(layer,block) pairs removed")
# ax.set_ylabel("cosine similarity")
# ax.set_title("cosine similarities with removing one LN layer")
# fig1.savefig(os.path.join(path,'LN_before_ablation'))

# ln_after_sims = [0.98986,0.971098,0.99364,0.98223,0.480766,0.746054,0.603211,0.956122,0.90858,0.98614,0.55668,0.57819,0.00901,0.69654,0.71322,0.61606,0.574877,0.64486,0.763875,0.636391]
# ln_after_x = list(range(len(ln_after_sims)))
# fig2 = plt.figure(figsize=(12,10))
# ax = fig2.add_subplot(1,1,1)
# ax.bar(ln_after_x,ln_after_sims,tick_label=ln_labels)
# ax.set_xlabel("(layer,block) pairs removed")
# ax.set_ylabel("cosine similarity")
# ax.set_title("cosine similarities with removing one LN layer")
# fig2.savefig(os.path.join(path,'LN_after_ablation'))



exit()

ln_both_sims = [0.945670,0.637312,0.990499,0.653348,0.403636,0.551515,0.480766,0.945670,0.585094,0.585094,0.409299,0.495281,0.683659,0.683659,0.570819,0.370292,0.68365,0.68365,0.540240,0.498782]
ln_both_x = list(range(len(ln_both_sims)))
fig2 = plt.figure(figsize=(12,10))
ax = fig2.add_subplot(1,1,1)
ax.bar(ln_both_x,ln_both_sims,tick_label=ln_labels)
ax.set_xlabel("(layer,block) pairs removed")
ax.set_ylabel("cosine similarity")
ax.set_title("cosine similarities with removing one LN layer")
fig2.savefig(os.path.join(path,'LN_both_ablation'))



