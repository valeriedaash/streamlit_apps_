import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from io import BytesIO

st.write("## Image compresser")

url = st.sidebar.text_input('Give url of your image', 'https://thumb.tildacdn.com/tild6662-3633-4436-b138-613038646330/-/format/webp/790.jpg')
image1 = io.imread(url)
image = image1.astype(np.float32)
percent = st.sidebar.slider('What percentage of quality should be left?', 0, 100, 50)
channels = []
k = int(min(image.shape[0], image.shape[1]) * percent / 100)  

for i in range(image.shape[2]):
    U, sing_vals, V = np.linalg.svd(image[:,:,i])
    
    sigma = np.zeros((image.shape[0], image.shape[1]))
    np.fill_diagonal(sigma, sing_vals)
    
    trunc_U = U[:, :k]
    trunc_sigma = sigma[:k, :k]
    trunc_V = V[:k, :]

    channel = trunc_U @ trunc_sigma @ trunc_V
    channels.append(channel)

channels = np.array(channels)

compressed_image = np.stack(channels, axis=-1)
compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
compressed_image = compressed_image.reshape(886, 700, -1)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.imshow(image)
ax1.set_title('Исходное изображение')

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.imshow(compressed_image)
ax2.set_title(f'Изображение на top {k} сингулярных чисел');

st.pyplot(fig1)
st.pyplot(fig2)

st.write(f'# Image is compressed to {percent} % quality')

download_button = st.download_button(
    label="Download Compressed Image",
    data=BytesIO(compressed_image.tobytes()),
    file_name=f"compressed_image_{percent}.png",
    key=f"download_button_{percent}"
)
