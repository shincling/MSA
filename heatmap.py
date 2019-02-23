#coding=utf8
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def draw_map(data_path,mask_seq):
    # data_path='visions/'
    shutil.rmtree(data_path)
    os.mkdir(data_path)
    mask_seq=mask_seq[:,:,:,0]
    frames=mask_seq.shape[0]
    for iii in range(frames):
        f, ax1 = plt.subplots(figsize=(4, 4),nrows=1 )
        # cmap用cubehelix map颜色
        # cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
        # sns.heatmap(mask_seq[4], linewidths=0.05, ax=ax1, vmax=1, vmin=0, cmap=cmap)
        # sns.heatmap(mask_seq[iii], annot=False, linewidths=0.05, ax=ax1, vmax=0.01, vmin=0, cmap='rainbow')
        sns.heatmap(mask_seq[iii], annot=False, linewidths=0.05, ax=ax1, cmap='rainbow')
        ax1.set_title('cubehelix map')
        ax1.set_xlabel('')
        ax1.set_xticklabels([])  # 设置x轴图例为空值
        ax1.set_ylabel('kind')

        f.savefig(data_path+'sns_heatmap_normal_{}.jpg'.format(iii+1), bbox_inches='tight')
