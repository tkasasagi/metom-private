from io import BytesIO

from PIL import Image
import requests
import torch
from transformers import AutoModel, AutoProcessor

repo_name = "."  # Use local directory
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

image1 = Image.open("examples/example1_4E00.jpg").convert("RGB")  # Use local example image
image_array1 = processor(images=image1, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch_dtype)
with torch.inference_mode():
    print(model.get_predictions(image_array1))  # Returns the prediction label
# ['一']

image2 = Image.open("examples/example2_5B9A.jpg").convert("RGB")  # Use local example image
image3 = Image.open("examples/example3_5009.jpg").convert("RGB")  # Use local example image
image_array2 = processor(images=[image2, image3], return_tensors="pt")["pixel_values"].to(device=device, dtype=torch_dtype)
with torch.inference_mode():
    print(model.get_topk_labels(image_array2))  # Returns top-k prediction labels (label only)
    # [['定', '芝', '乏', '淀', '実'], ['倉', '衾', '斜', '会', '急']]
    print(model.get_topk_labels(image_array2, k=3, return_probs=True))  # Returns prediction top-k labels (label with probability)
    # [[('定', 0.9979104399681091), ('芝', 0.0002953427319880575), ('乏', 0.00012814522779081017)], [('倉', 0.9862521290779114), ('衾', 0.0005956474924460053), ('斜', 0.00039981433656066656)]]
