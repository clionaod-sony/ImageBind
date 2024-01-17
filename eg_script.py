from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# text_list = ["A dog.", "A car", "A bird", "Vanilla"]
# image_paths = [
#     ".assets/dog_image.jpg",
#     ".assets/car_image.jpg",
#     ".assets/bird_image.jpg",
#     ".assets/vanilla_image.jpg",
# ]

image_paths = glob.glob("../olfactory/datasets/toy_dataset/images/*/*/*.jpg")
cluster_maps = {i.split("/")[-3]: [] for i in image_paths}
for i in image_paths:
    cluster_maps[i.split("/")[-3]].append(i)
stimulus_maps = {i.split("/")[-1]: i for i in image_paths}

text_list = list(stimulus_maps.keys())
text_list.sort()

image_paths = [stimulus_maps[i] for i in text_list]
text_list = [i.split(".")[0][:-1] for i in text_list]

df = pd.DataFrame(columns=["image_paths", "text", "cluster"])
df["image_paths"] = image_paths
df["text"] = text_list
for cluster, stimlist in cluster_maps.items():
    for stim in stimlist:
        df.loc[df.image_paths == stim, "cluster"] = cluster
df = df.sort_values(["cluster", "text"])

text_list = []

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(df.text, device),
    ModalityType.VISION: data.load_and_transform_vision_data(df.image_paths, device),
    # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

res_df = pd.DataFrame(embeddings[ModalityType.VISION].cpu())
res_df = pd.concat([res_df, pd.DataFrame(embeddings[ModalityType.TEXT].cpu())])
res_df.index = [i.split("/")[-1] for i in df.image_paths] + df.text.to_list()
corrcoef = res_df.T.corr()

cross_dom = corrcoef.iloc[:45, 45:]

sns.heatmap(cross_dom)
plt.show()


print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

print()

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
