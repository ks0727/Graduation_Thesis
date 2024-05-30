import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch.nn as nn
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"../../"))
from Make_Input.code import Make_Input

def main():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    datasets = load_dataset("hf-internal-testing/example-documents", split="test")

    #image = datasets["image"][0]
    make_input = Make_Input.Make_input(text='A',font_size=500)
    image,_ = make_input.create_image()
    task_prompt = "<s_iitcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    feature_map = []

    def hook(model,input,output):
        feature_map.append(output[0].data.cpu())
    
    def visualize_map(map,fig_name):
        abs_path = os.path.dirname(__file__)
        result_path = "../result"
        path = os.path.join(abs_path,result_path)
        path = os.path.join(path,fig_name)

        map = (map+1)/2
        plt.figure(figsize=(10,10))
        plt.imshow(map,cmap='gray')
        plt.colorbar()
        plt.savefig(path)
    
    model.encoder.embeddings.patch_embeddings.register_forward_hook(hook)
    output = model.encoder(pixel_values.to(device))
    del output
    torch.cuda.empty_cache()
    fmap = feature_map[0].squeeze()

    for i in range(fmap.shape[1]):
        map = fmap[:,i]
        map = map.reshape(640,480)
        fig_name = f"{i+1}th_feature_map_by_embedding"
        visualize_map(map,fig_name)


if __name__ == '__main__':
    main()