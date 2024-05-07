import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os
from typing import Any,List,Tuple

class Attention_map:
    def __init__(self,attention:torch.Tensor, num_head:int, save_path:str, h_image:int, w_image:int) -> None:
        self.attention = attention
        self.num_head = num_head
        self.save_path = save_path
        self.h_image = h_image
        self.w_image = w_image

    def get_attention_map(self)->npt.NDArray:
        h,w = self.h_image//self.num_head, self.w_image//self.num_head #(h,w) == (height of the feature map, width of the feature map)
        window_size = 10 

        attention_np = self.attention.to('cpu').detach().numpy().copy() #convert it to numpy array
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
        attn_map_name = "num_head_" + str(self.num_head) + "_" + str(i+1) + "th.png"
        path = os.path.join(self.save_path,attn_map_name)
        plt.figure(figsize=(10,10))
        plt.imshow(attention_map_i,vmin=0,vmax=0.8,interpolation='nearest')
        plt.xlabel('width')
        plt.ylabel('height')
        plt.title(f'{i+1}th/{self.num_head} head attention map <pretrained>')
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

    def get_stat_all_stages(self,outputs)->Tuple[List[float],List[float]]:
        attention_avgs = []
        attention_vars = []

        for i in range(len(outputs.encoder_attentions)):
            attn_maps = self.get_attn_map(outputs.encoder_attentions[i],outputs.encoder_attentions[i].shape[1],save_path=self.save_path)
            avg, var = self.stat_attn_maps(attn_maps)
            attention_avgs.append(avg)
            attention_vars.append(var)
        
        return attention_avgs,attention_vars


    def get_mean_bar_graph(self,mean:List[float])->None:
        stage = list(range(len(mean)))
        label = ['stage'+str(i+1) for i in range(len(mean))]
        plt.figure(figsize=(10,10))
        plt.bar(stage,mean,tick_label=label)
        plt.xlabel("stages")
        plt.ylabel("mean")
        plt.title("mean values at each stage")
        graph_name = "mean_at_each_stage_donut_base"
        path = os.path.join(self.save_path,graph_name)
        plt.show()
        plt.savefig(path)

    def get_variance_bar_graph(self,variance:List[float])->None:
        stage = list(range(len(variance)))
        label = ['stage'+str(i+1) for i in range(len(variance))]
        plt.figure(figsize=(10,10))
        plt.bar(stage,variance,tick_label=label)
        plt.xlabel("stages")
        plt.ylabel("variance")
        plt.title("variance values at each stage")
        graph_name = "variance_at_each_stage_donut_base"
        path = os.path.join(self.save_path,graph_name)
        plt.show()
        plt.savefig(path)
