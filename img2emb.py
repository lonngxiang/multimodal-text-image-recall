"""
缩略图向量化
"""
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 加载媒资内容
kkk_all = np.load(r"D:****容1.npy")
aids = list(set([i[0] for i in kkk_all.tolist()]))

# 加载模型
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 计算缩略图向量
aidss=[]
imgs=[]
for i in tqdm(aids):

    try:

        aidss.append(i)
        image1 = preprocess(Image.open(r"D:****图\{}.jpg".format(i))).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image1)
            imgs.append(image_features)
            print(type(image_features))

    except Exception as e:
        print(e)
        print("####")
        aidss.pop()
        pass

# 保存
np.save(r"D:*****dss.npy", aidss)
np.save(r"D:\****s_embs.npy", imgs)

