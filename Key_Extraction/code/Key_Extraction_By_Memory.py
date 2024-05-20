from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
import json
import random

#function to convert torch.Tensor() to ndarray
def to_np(x:torch.Tensor):
    x = x.to('cpu').detach().numpy().copy()
    return x

def print_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")


def main(args)->int:
    model_path,dataset_path = args.pretrained_model_path, args.dataset_path
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.config.output_hidden_states = True #changed the config to output the hidden states
    model.to(device)
    model.eval()
    
    task_prompt = "<s_iitcdip>"
    print(model.encoder)
    dataset = load_dataset(dataset_path, split="train")

    top_k_data = [] #the list to have the information (similarity,dataset_index,patch_index)
    dims_for_analysis = args.dims_for_analysis
    max_dim = 512*2**(args.stage_for_analysis) #max dimension of the key to analyze
    save = []
    def hook(model,input,output):
        # print(output.is_cuda)
        # print(output.shape)
        # save.append(output.detach())
        # save.append(output.to('cpu'))
        save.append(output.data.cpu())
    
    encoder = model.encoder
    encoder.config.output_hidden_states = True
    encoder.encoder.layers[args.stage_for_analysis].blocks[args.block_for_analysis].intermediate.register_forward_hook(hook)

    if dims_for_analysis is None:
        dims_for_analysis = random.randint(0,max_dim-1) #decide the dimension of the weight to analyze if it was not specified
    
    assert dims_for_analysis is not None
    
    for idx in tqdm(range(args.max_images)):
        with torch.no_grad():
            image = dataset[idx]["image"]
            pixel_values = processor(image, return_tensors="pt").pixel_values
            _ = model.encoder(pixel_values.to(device))
            calculated_sim = save[-1]
            calculated_sim = calculated_sim.squeeze()[:,dims_for_analysis]
            calculated_sim = to_np(calculated_sim)
            sorted_idx = np.argsort(-calculated_sim)
            sorted_idx = sorted_idx[:args.top_k]
            for i in range(args.top_k):
                top_k_data.append((calculated_sim[sorted_idx[i]],idx,sorted_idx[i]))
            top_k_data = sorted(top_k_data,reverse=True)[:args.top_k]
            save.pop(-1)

    result_path = os.path.join(args.result_path,f"{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top")
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    txt_file_name = f"{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top.txt"
    json_file_name = f"{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top.json"
    
    path = os.path.join(os.path.dirname(__file__),result_path)
    txt_path = os.path.join(path,txt_file_name)
    json_path = os.path.join(path,json_file_name)
    
    try:
        dict_top_k = {}
        with open(txt_path,"w") as f:
            f.write(f"CONFIG : images : {args.max_images}, stage : {args.stage_for_analysis}, block : {args.block_for_analysis}, dimension : {dims_for_analysis}, top_k :{args.top_k}\n")
            for i in range(args.top_k):
                similarity,data_idx,patch_idx = top_k_data[i]
                f.write(f"rank{i}: ")
                f.write(f"similarity : {similarity}, dataset_idx : {data_idx}, path_idx : {patch_idx}\n")
                dict_top_k[str(i+1)] = {"rank":str(i+1),"similarity":str(similarity),"data_index":str(data_idx),"patch_index":str(patch_idx)}
        with open(json_path,"w") as f:
            json.dump(dict_top_k,f,indent=2)
    except:
        with open("result.txt","w") as f:
            for i in range(args.top_k):
                similarity,data_idx,patch_idx = top_k_data[i]
                f.write(f"rank{i}: ")
                f.write(f"similarity : {similarity}, dataset_idx : {data_idx}, path_idx : {patch_idx}\n")
        raise AssertionError("couldn't open the given file path")
    
    return dims_for_analysis

def visualize_extracted_patches(args,dims_for_analysis:int)->None:
    result_path = args.result_path
    json_file_name = f"{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top.json"
    result_path = os.path.join(result_path,f"{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top")
    path = os.path.join(os.path.dirname(__file__),result_path)
    path = os.path.join(path,json_file_name)
    with open(path,"r") as f:
        dict_top_k = json.load(f)
    
    processor = DonutProcessor.from_pretrained(args.pretrained_model_path)
    dataset = load_dataset(args.dataset_path, split="train")

    fig_save_dir = os.path.join(args.result_path,f"{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top")
    fig_save_dir = os.path.join(os.path.dirname(__file__),fig_save_dir)
        
    if not os.path.isdir(fig_save_dir):
        os.makedirs(fig_save_dir)
    
    for i in range(30):
        data_idx,patch_idx = int(dict_top_k[str(i+1)]['data_index']),int(dict_top_k[str(i+1)]["patch_index"])
        data = dataset[data_idx]
        image = data["image"]
        
        pixel_values = processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        pixel_values = pixel_values.permute(1,2,0)
        pixel_values = pixel_values*0.5+0.5
        
        patch_scale = args.stage_for_analysis
        pixel_h = patch_idx//(480//2**patch_scale)
        pixel_w = patch_idx%(480//2**patch_scale)
        
        patch_size = 2**(2+patch_scale)
        patch_extracted = torch.zeros((patch_size,patch_size,3))
        patch_extracted[:,:,:] = pixel_values[pixel_h*patch_size : pixel_h*patch_size + patch_size, pixel_w * patch_size : pixel_w*patch_size+patch_size,:]
        
        plt.imshow(patch_extracted)
        fig_path = os.path.join(os.path.dirname(__file__),fig_save_dir)
        fig_name = f'{args.max_images}_images_{args.stage_for_analysis}_stage_{args.block_for_analysis}_block_{dims_for_analysis}_dim_{args.top_k}_top_' + str(i+1)+'th'
        plt.savefig(os.path.join(fig_path,fig_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path',default="naver-clova-ix/donut-base",type=str,
                        help='path to pretrained model')
    parser.add_argument('--dataset_path',default="naver-clova-ix/synthdog-ja",type=str,
                        help='path to dataset')
    parser.add_argument('--result_path',default="../result",type=str)
    parser.add_argument('--top_k',default=50,type=int,
                        help='how many top patches to extract')
    parser.add_argument('--max_images',default=1000,type=int,
                        help='how many images you want to use for analysis')
    parser.add_argument('--stage_for_analysis',default=0,type=int,
                        help='choose from 0 ~ 3')
    parser.add_argument('--block_for_analysis',default=0,type=int,
                        help='block to choose. please choose from either 0 or 1')
    parser.add_argument('--dims_for_analysis',default=None,type=int,
                        help='dimensions to get triggered examples')
    args = parser.parse_args()

    dim = main(args)
    visualize_extracted_patches(args,dim)