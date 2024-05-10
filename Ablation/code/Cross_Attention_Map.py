from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class CrossAttentionMap:
    def __init__(self,cross_attns,path) -> None:
        self.cross_attns = cross_attns
        self.path = path

    def get_cross_attn_maps(self,image=None,withImage=False,output_sequence=None,processor=None):
        output_sequence = output_sequence.to('cpu').detach().numpy().copy()
        for i in range(len(self.cross_attns)):
            cross_attn_ith_word = self.cross_attns[i][0]
            cross_attn_ith_word = cross_attn_ith_word.squeeze()
            cross_attn_ith_word_np = cross_attn_ith_word.to('cpu').detach().numpy().copy()
            cross_attn_ith_word_np = np.mean(cross_attn_ith_word_np,axis=0)
            cross_attn_map_ith_word = np.zeros((80,60))
            cross_attn_map_ith_word[:][:] = cross_attn_ith_word_np.reshape((80,60))
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)
            ax.imshow(cross_attn_map_ith_word,interpolation='nearest',aspect='auto',alpha=1.0)
            ax.set_xlabel('width')
            ax.set_ylabel('height')
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if withImage:
                assert image is not None , "please provide the image to show with"
                ax.imshow(image,extent=[*xlim,*ylim],aspect='auto',alpha=0.8)
            if output_sequence is None:
                ax.set_title(f'{i+1}th word cross attention map <Dence_layer_Ablation>')
            else:
                word = processor.decode(output_sequence[i])
                ax.set_title(f'{i+1}th word: "{word}" cross attention map <Dence_layer_Ablation>')
            fig.colorbar(mappable=None)
            save_path = os.path.join(self.path,f'{i+1}th_word_Before_Ablation_ablation_cross_attn_map')
            fig.savefig(save_path)
    