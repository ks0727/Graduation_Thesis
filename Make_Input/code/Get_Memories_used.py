from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from Make_Input import Make_input
import os
import argparse
import json
import random

#function to convert torch.Tensor() to ndarray
def to_np(x:torch.Tensor):
    x = x.to('cpu').detach().numpy().copy()
    return x

def main()->int:
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    #task_prompt = "<s_iitcdip>"

    top_memories = []
    save = []
    
    def hook(model,input,output):
        output_cpu = output.data.cpu()
        output_cpu = output_cpu.squeeze()
        #top_memories.append(output)
        save.append(output_cpu)
    
    encoder = model.encoder
    encoder.config.output_hidden_states = True
    
    for i in range(4):
        jmax = 14 if i == 2 else 2
        for j in range(jmax):
            encoder.encoder.layers[i].blocks[j].intermediate.register_forward_hook(hook)
    input_text = 'A'
    make_input = Make_input(text=input_text,font_size=90,text_pos=(0,256))
    image,_ = make_input.create_image()
    
    abs_path = os.path.dirname(__file__)
    result_path = "../result/memories_used"
    image_name = "test.png"
    path = os.path.join(abs_path,result_path)
    image_path = os.path.join(path,image_name)
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    _ = model.encoder(pixel_values.to(device))
    result_dict = {}
    
    mx_memories = []
    for si in range(len(save)):
        output_i = save[si]
        memories = output_i.sum(dim=0)
        mx_idx = torch.argsort(-memories)
        mx_idx = mx_idx.to('cpu')[:30].tolist()
        mx_memories.append(mx_idx)
        
    result_dict[input_text] = mx_memories
    json_file_name = f"input_{input_text}.json"
    txt_file_name = f"input_{input_text}.txt"
    blocks = [2,2,14,2]
    dir_to_save = f"{input_text}_memories"
    
    if not os.path.isdir(os.path.join(path,dir_to_save)):
        os.makedirs(os.path.join(path,dir_to_save))
    save_path = os.path.join(path,dir_to_save)
    
    with open(os.path.join(save_path,txt_file_name),"w") as f:
        now_idx = 0
        f.write(f"input text : {input_text}\n")
        for stage in range(4):
            for block in range(blocks[stage]):
                f.write(f"stage {stage}, block {block} memories : {mx_memories[now_idx]}\n")
                now_idx += 1
            
    with open(os.path.join(save_path,json_file_name),"w") as f:
        json.dump(result_dict,f,indent=1)

if __name__ == '__main__':
    main()