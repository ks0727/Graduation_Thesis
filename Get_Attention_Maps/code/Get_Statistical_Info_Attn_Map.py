from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import os
from typing import List, Tuple

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.config.output_attentions = True # I changed this config so that I can get the attention
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("naver-clova-ix/cord-v2", split="test")

task_prompt = "<s_iitcdip>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

def get_attn_map(attention :torch.Tensor,num_head : int, save_path : str, ouput_attention:bool = False)->List[npt.NDArray]:
    global h_image, w_image
    h,w = h_image//num_head, w_image//num_head #(h,w) == (height of the feature map, width of the feature map)
    window_size = 10 

    attention_np = attention.to('cpu').detach().numpy().copy() #convert it to numpy array
    attention_np = np.transpose(attention_np,(1,0,2,3)) #(3072,4,100,100) -> (4,3072,100,100)
    attention_maps = []
    for i in range(attention_np.shape[0]):
        attention_np_i = attention_np[i]
        
        attention_np_i = np.reshape(attention_np_i,(h//window_size,w//window_size,-1,window_size,window_size)) # (3072,100,100) -> (64,48,100,10,10)
        
        attention_np_i = np.transpose(attention_np_i,(2,0,1,3,4)) #(64,48,100,10,10)->(100,64,48,10,10)
        
        attention_np_i_new = attention_np_i[0]#(64,48,10,10)
        attention_map_i = np.zeros((h,w)) #(640,480)
        for a in range(0,h,window_size):
            for b in  range(0,w,window_size):
                attention_map_i[a:a+window_size,b:b+window_size] = attention_np_i_new[a//window_size][b//window_size] #put puthes into the map

        attention_maps.append(attention_map_i)

    return attention_maps

def stat_attn_maps(att_maps : List[npt.NDArray])->Tuple[float,float]:
    attn_means = []
    attn_vars = []
    for attn_map in att_maps:
        attn_mean = np.mean(attn_map)
        attn_var = np.var(attn_map)
        attn_means.append(attn_mean)
        attn_vars.append(attn_var)
    avg = sum(attn_means)/len(attn_means)
    var = sum(attn_vars)/len(attn_vars)
    return avg,var

save_path = "./result/donut_base"
means = np.zeros((len(dataset),4))
vars = np.zeros((len(dataset),4))

for i,data in tqdm(enumerate(dataset),total=len(dataset)):
    image = data["image"]
    pixel_values = processor(image, return_tensors="pt").pixel_values
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids = decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True
    )
    avgs = []
    variances = []
    h_image, w_image = pixel_values.size(2),pixel_values.size(3)
    for j in range(len(outputs.encoder_attentions)):
        attn_maps = get_attn_map(outputs.encoder_attentions[j],outputs.encoder_attentions[j].shape[1],save_path=save_path)
        avg, var = stat_attn_maps(attn_maps)
        avgs.append(avg)
        variances.append(var)
    
    means[i] = np.array(avgs)
    vars[i] = np.array(variances)


print(vars)
print(vars.shape)
text_file_name = "static_information_attn_map_100_images.txt"
path = os.path.join(save_path,text_file_name)

with open(path,"w") as f:
    f.write("mean values with 100 images\n")
    f.write(str(means))
    f.write("\n")
    f.write("variance value with 100 images\n")
    f.write(str(vars))
    f.write("\n")
    
mean_list = np.mean(means,axis=0)
var_list = np.var(vars,axis=0)
stage_list = np.arange(mean_list.shape[0])

def get_mean_bar_graph(stage_list,mean_list,save_path:str)->None:
    label = ["stage"+str(i+1) for i in range(stage_list.shape[0])]
    plt.figure(figsize=(10,10))
    plt.bar(stage_list,mean_list,tick_label=label)
    plt.xlabel("stages")
    plt.ylabel("mean")
    plt.title("mean values at each stage")
    graph_name = "mean_with_100images_at_each_stage_donut_base"
    path = os.path.join(save_path,graph_name)
    plt.show()
    plt.savefig(path)

def get_variance_bar_graph(stage_list,var_list,save_path:str)->None:
    label = ['stage'+str(i+1) for i in range(stage_list.shape[0])]
    plt.figure(figsize=(10,10))
    plt.bar(stage_list,var_list,tick_label=label)
    plt.xlabel("stages")
    plt.ylabel("variance")
    plt.title("variance values at each stage")
    graph_name = "variance_with_100images_at_each_stage_donut_base"
    path = os.path.join(save_path,graph_name)
    plt.show()
    plt.savefig(path)

get_mean_bar_graph(stage_list,mean_list,save_path)
get_variance_bar_graph(stage_list,var_list,save_path)