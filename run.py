"""
缩略图向量语义召回
"""
import numpy as np
from PIL import Image
import requests
import random
from flask import Flask
from flask import render_template, request
import torch
import clip


# 加载数据
kkk_dict_all = np.load(r"D:***.npy", allow_pickle=True).item()

image1 = np.load(r"D:***mbs.npy", allow_pickle=True)
aidss = np.load(r"D:****ss.npy")

image = torch.Tensor([(item / item.norm(dim=-1, keepdim=True)).cpu().numpy() for item in image1])

# 加载模型
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def build_plot():

  query = request.args.get('query')
  print(query)

  # query翻译成英文
  data22 = {
      "i": query,
      "from": "AUTO",
      "to": "AUTO",
      "smartresult": "dict",
      "client": "fanyideskweb",
      "doctype": "json",
      "version": "2.1",
      "keyfrom": "fanyi.web",
      "action": "FY_BY_REALTIME",
      "typoResult": "false"
  }
  response = requests.post("http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule", data=data22).json()
  bb = response["translateResult"][0][0]["tgt"]

  # 文本向量化计算
  text1 = clip.tokenize([bb]).to(device)

  with torch.no_grad():
      text_features = model.encode_text(text1)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      # 文本与图形相似度计算召回
      similarities = (image @ text_features.T).squeeze(1)
      best_photo_idx = np.argsort(similarities[:, 0].numpy())[::-1]
      rank_results = [aidss[i] for i in best_photo_idx[:30]]
      # 召回结果封装
      titles1 = []
      pics1 = []
      for j in rank_results:
          titles1.append(kkk_dict_all[j][0])
          pics1.append(kkk_dict_all[j][1])
      print(titles1, pics1)

  # 中控返回结果
  url1 = "http:*****48&query={}&device_id=test&ver=3.0&user_id={}".format(
      query, random.randint(323, 98080983))
  data11 = {
      'json': '{"device_status":*****
  }
  r = requests.get(url1, params=data11).json()
  # 中控结果封装
  try:
      global titles, pics
      previously_datas = r["data"]["json"]["results"][0]["resources"]["response"]["thirdData"]["previouslyMovie"]
      aids = []
      titles = []
      pics = []
      if previously_datas:
          for i in previously_datas:
              aids.append(i["provider"] + i["cover_id"])
              titles.append(i["resource_name"])
              pics.append(i["thumb"])

  except:
      titles=[]
      pics=[]
      pass

  if titles:
      return render_template('display1.html', query=query, translate=bb, lis1=titles[:10], lis2=pics[:10], lis3=titles1, lis4=pics1)
  else:
      return render_template('display2.html', query=query,  translate=bb, lis3=titles1, lis4=pics1)


if __name__ == '__main__':
  app.run("0.0.0.0", 6600, debug=True, threaded=True)
