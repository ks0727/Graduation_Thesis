import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
from typing import List, Tuple

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.config.output_attentions = True # I changed this config so that I can get the attention
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
image.show()

task_prompt = "<s_iitcdip>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

#(h_image,w_image) : (height of the image, width of the image)
h_image, w_image = pixel_values.size(2),pixel_values.size(3)

print(f'the length of the encoder_attentions: {len(outputs.encoder_attentions)}')

# display the shape of the encoder attentions of the layer
for i in range(len(outputs.encoder_attentions)):
    enc_attn = outputs.encoder_attentions[i]
    print(f'layer :{i} the shape of the encoder attention is {enc_attn.shape}')

"""
the shape of the tensor
first layer  : (3072,4,100,100)
second layer : (768,8,100,100)
third layer  : (192,16,100,100)
fourth layer : (48,32,100,100)
ex) first layer
    original height and width of the image (H,W) -> (2560,1920)
    3072 -> 64*48
    num_head = 4
    640 -> 2560 // 4
    480 -> 1920 // 4
    100 -> window_size * window_size
    100 -> window_size * window_size
"""

# this function gives you attention maps
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

        # plot and save the attention map
        attn_map_name = "num_head_" + str(num_head) + "_" + str(i+1) + "th.png"
        path = os.path.join(save_path,attn_map_name)
        plt.figure(figsize=(10,10))
        plt.imshow(attention_map_i,vmin=0,vmax=0.8,interpolation='nearest')
        plt.xlabel('width')
        plt.ylabel('height')
        plt.title(f'{i+1}th/{num_head} head attention map <pretrained>')
        plt.colorbar()
        plt.savefig(path)

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
attention_avgs = []
attention_vars = []

for i in range(len(outputs.encoder_attentions)):
    attn_maps = get_attn_map(outputs.encoder_attentions[i],outputs.encoder_attentions[i].shape[1],save_path=save_path)
    avg, var = stat_attn_maps(attn_maps)
    attention_avgs.append(avg)
    attention_vars.append(var)

with open("pre_attention_avg_result.txt","w") as f:
    s = [str(x) for x in attention_avgs]
    t = [str(x) for x in attention_vars]
    s = " ".join(s)
    t = " ".join(t)
    s += '\n'
    t += '\n'
    f.write("mean:  ")
    f.write(s)
    f.write("variance:  ")
    f.write(t)


def get_mean_bar_graph(mean:List[float],save_path:str)->None:
    stage = list(range(len(mean)))
    label = ['stage'+str(i+1) for i in range(len(mean))]
    plt.figure(figsize=(10,10))
    plt.bar(stage,mean,tick_label=label)
    plt.xlabel("stages")
    plt.ylabel("mean")
    plt.title("mean values at each stage")
    graph_name = "mean_at_each_stage_donut_base"
    path = os.path.join(save_path,graph_name)
    plt.show()
    plt.savefig(path)

def get_variance_bar_graph(variance:List[float],save_path:str)->None:
    stage = list(range(len(variance)))
    label = ['stage'+str(i+1) for i in range(len(variance))]
    plt.figure(figsize=(10,10))
    plt.bar(stage,variance,tick_label=label)
    plt.xlabel("stages")
    plt.ylabel("variance")
    plt.title("variance values at each stage")
    graph_name = "variance_at_each_stage_donut_base"
    path = os.path.join(save_path,graph_name)
    plt.show()
    plt.savefig(path)

get_variance_bar_graph(attention_vars,save_path)
get_mean_bar_graph(attention_avgs,save_path)

decoded_results = processor.tokenizer.batch_decode(outputs.sequences)
#sequence = processor.batch_decode(outputs.sequences)[0]
#sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
#sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print("-------------------------------------------")
print(decoded_results)
#print("output : ", processor.token2json(sequence))
print("-------------------------------------------")