---
license: apache-2.0
---
# Metom (めとむ)

The **Metom** is a Vision Transformer (ViT) based **Kuzushiji** classifier.
The model takes an image with one character and returns what the character is.
**This model is not an official SakanaAI product and is for research / educational purposes only.**

**めとむ**は Vision Transformer (ViT) ベースの**くずし字**分類器です。
モデルは1文字が写った画像を受け取り、その文字がどの文字であるかを返します。
**本モデルはSakanaAIの公式製品ではありません。研究・教育目的のみに利用してください。**

*Japanese section follows English section (日本語セクションは英語セクションの後に続きます。)*

--------------------------------------------------------------------------------

This model was trained by using [日本古典籍くずし字データセット](http://codh.rois.ac.jp/char-shape/book/).
This dataset contains 1,086,326 characters in 4,328 types of Kuzushiji.
However, we used only 2,703 types of characters that appeared at least 5 times in the dataset.

The dataset was split into train, validation, and test subsets in a ratio of 3:1:1.
As a result, the train subset contained 649,932 characters, the validation subset contained 216,644 characters, and the test subset contained 216,645 characters.

The model was trained on the train subset, and hyperparameters were tuned based on the performance on the validation subset.
The final evaluation on the test subset yielded a micro accuracy of 0.9722 and a macro accuracy of 0.8354.

## Usage
Please see also [Google Colab Notebook](https://colab.research.google.com/drive/1jFMZENoTjjum3qlBxV0Q5dTxmpCvqlpf?usp=sharing).
1. Install dependencies (Not required on Google Colab)
```sh
python -m pip install einops torch torchvision transformers

# Optional (This is also required on Google Colab if you want to use FlashAttention-2)
pip install flash-attn --no-build-isolation
```

2. Run the following code
```python
from io import BytesIO

from PIL import Image
import requests
import torch
from transformers import AutoModel, AutoProcessor

repo_name = "SakanaAI/Metom"
device = "cuda"
torch_dtype = torch.float32  # This can also set `torch.float16` or `torch.bfloat16`

def get_image(image_url: str) -> Image.Image:
    return Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

processor = AutoProcessor.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    repo_name,
    torch_dtype=torch_dtype,
    _attn_implementation="eager",  # This can also set `"sdpa"` or `"flash_attention_2"`
    trust_remote_code=True
).to(device=device)

image1 = get_image("https://huggingface.co/SakanaAI/Metom/resolve/main/examples/example1_4E00.jpg")  # An example image
image_array1 = processor(images=image1, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch_dtype)
with torch.inference_mode():
    print(model.get_predictions(image_array1))  # Returns the prediction label
# ['一']

image2 = get_image("https://huggingface.co/SakanaAI/Metom/resolve/main/examples/example2_5B9A.jpg")  # An example image
image3 = get_image("https://huggingface.co/SakanaAI/Metom/resolve/main/examples/example3_5009.jpg")  # An example image
image_array2 = processor(images=[image2, image3], return_tensors="pt")["pixel_values"].to(device=device, dtype=torch_dtype)
with torch.inference_mode():
    print(model.get_topk_labels(image_array2))  # Returns top-k prediction labels (label only)
    # [['定', '芝', '乏', '淀', '実'], ['倉', '衾', '斜', '会', '急']]
    print(model.get_topk_labels(image_array2, k=3, return_probs=True))  # Returns prediction top-k labels (label with probability)
    # [[('定', 0.9979104399681091), ('芝', 0.0002953427319880575), ('乏', 0.00012814522779081017)], [('倉', 0.9862521290779114), ('衾', 0.0005956474924460053), ('斜', 0.00039981433656066656)]]
```

## Citation
```bibtex
@misc{Metom,
    url    = {[https://huggingface.co/SakanaAI/Metom](https://huggingface.co/SakanaAI/Metom)},
    title  = {Metom},
    author = {Imajuku, Yuki and Clanuwat, Tarin}
}
```

--------------------------------------------------------------------------------

本モデルは[日本古典籍くずし字データセット](http://codh.rois.ac.jp/char-shape/book/)を用いて訓練されました。
このデータセットは4,328種1,086,326枚のくずし字画像が含まれています。
ですが、データセット中に最低5回以上出現する2,703種類の文字のみを利用しました。

データセットは訓練、検証、テストの3つのセットに、比率が3:1:1となるように分割されました。
その結果、訓練セットは649,932枚、検証セットは216,644枚、テストセットは216,645枚、画像が含まれました。

本モデルは訓練セットのみを用いて学習され、検証セットにおける性能を見ながらハイパーパラメータを調整しました。
最終的にテストセットにおける評価の結果、216,645枚全体の正解率は0.9722となり、2,703種類のクラス別正解率の平均は0.8354となりました。

## 使用方法
[Google Colab Notebook](https://colab.research.google.com/drive/1jFMZENoTjjum3qlBxV0Q5dTxmpCvqlpf?usp=sharing)もご確認ください。
1. 依存ライブラリをインストールする (Google Colabを使う場合は不要)
```sh
python -m pip install einops torch torchvision transformers

# 任意 (FlashAttention-2を使いたい場合はGoogle Colabを使う時でも必要)
pip install flash-attn --no-build-isolation
```

2. 以下のコードを実行する
```python
from io import BytesIO

from PIL import Image
import requests
import torch
from transformers import AutoModel, AutoProcessor

repo_name = "SakanaAI/Metom"
device = "cuda"
torch_dtype = torch.float32  # `torch.float16` や `torch.bfloat16` も指定可能

def get_image(image_url: str) -> Image.Image:
    return Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

processor = AutoProcessor.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    repo_name,
    torch_dtype=torch_dtype,
    _attn_implementation="eager",  # `"sdpa"` や `"flash_attention_2"` も指定可能
    trust_remote_code=True
).to(device=device)

image1 = get_image("https://huggingface.co/SakanaAI/Metom/resolve/main/examples/example1_4E00.jpg")  # 画像例
image_array1 = processor(images=image1, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch_dtype)
with torch.inference_mode():
    print(model.get_predictions(image_array1))  # 予測ラベルを返す
# ['一']

image2 = get_image("https://huggingface.co/SakanaAI/Metom/resolve/main/examples/example2_5B9A.jpg")  # 画像例
image3 = get_image("https://huggingface.co/SakanaAI/Metom/resolve/main/examples/example3_5009.jpg")  # 画像例
image_array2 = processor(images=[image2, image3], return_tensors="pt")["pixel_values"].to(device=device, dtype=torch_dtype)
with torch.inference_mode():
    print(model.get_topk_labels(image_array2))  # 上位k件の予測ラベルを返す (ラベルのみ)
    # [['定', '芝', '乏', '淀', '実'], ['倉', '衾', '斜', '会', '急']]
    print(model.get_topk_labels(image_array2, k=3, return_probs=True))  # 上位k件の予測ラベルを返す (ラベルと確率)
    # [[('定', 0.9979104399681091), ('芝', 0.0002953427319880575), ('乏', 0.00012814522779081017)], [('倉', 0.9862521290779114), ('衾', 0.0005956474924460053), ('斜', 0.00039981433656066656)]]
```

## 引用
```bibtex
@misc{Metom,
    url    = {[https://huggingface.co/SakanaAI/Metom](https://huggingface.co/SakanaAI/Metom)},
    title  = {Metom},
    author = {Imajuku, Yuki and Clanuwat, Tarin}
}
```
