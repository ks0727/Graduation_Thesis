import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from Cross_Attention_Map import CrossAttentionMap
import os

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.config.output_attentions = True
model.config.output_hidden_states = True

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
task_prompt = "<s_iitcdip>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_img = pixel_values.squeeze()
pixel_img = pixel_img.permute(1,2,0)
pixel_img = pixel_img*0.5+0.5

save = []
def hook(module,input,output):
    save.append(output.detach())
exit()
encoder = model.encoder
encoder.config.output_hidden_states = True
encoder.encoder.layers[0].blocks[0].intermediate.register_forward_hook(hook)
output = encoder(pixel_values.to(device))
outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
    output_hidden_states=True,
)

rpath = '../result/CrossAttentionMaps/Before_Ablation'
path = os.path.join(os.path.dirname(__file__),rpath)
last = outputs.encoder_hidden_states[4]
print(last)
exit()
"""
this_file_name = os.path.basename(__file__)
this_file_name,_ = os.path.splitext(this_file_name)
cross_attns = outputs.cross_attentions
cross_attn_map = CrossAttentionMap(cross_attns=cross_attns,path=path)
cross_attn_map.get_cross_attn_maps(name=this_file_name,processor=processor,output_sequence=outputs.sequences[0])
"""
sequence = processor.batch_decode(outputs.sequences)[0]
print(sequence)
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(len(sequence))
print("-------------------------------------------")
print("output : ", processor.token2json(sequence))
print("-------------------------------------------")

