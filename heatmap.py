#coding=utf8
import seaborn as sns
import matplotlib.pyplot as plt

def draw_map(data_path,mask_seq):
    mask_seq=mask_seq[:,:,:,0]
    # frames,size0,size1=mask_seq.size()
    f, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=2)
    # cmap用cubehelix map颜色
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(mask_seq[4], linewidths=0.05, ax=ax1, vmax=1, vmin=0, cmap=cmap)
    ax1.set_title('cubehelix map')
    ax1.set_xlabel('')
    ax1.set_xticklabels([])  # 设置x轴图例为空值
    ax1.set_ylabel('kind')

    # cmap用matplotlib colormap
    sns.heatmap(mask_seq[5], linewidths=0.05, ax=ax2, vmax=1, vmin=0, cmap='rainbow')
    # rainbow为 matplotlib 的colormap名称
    ax2.set_title('matplotlib colormap')
    ax2.set_xlabel('region')
    ax2.set_ylabel('kind')

    f.savefig('visions/sns_heatmap_normal.jpg', bbox_inches='tight')
    return None
