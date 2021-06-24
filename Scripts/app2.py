import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import datetime
import skimage.io
import SessionState
import CLIPModel
from CLIPModel import *
import CFG
from CFG import *

@st.cache
def initialize_model():
    #weights_path_sketch = "Train Weights-CLIP/"
    weights_path_sketch = "./Scripts/Train Weights-CLIP/"
    sketchClassificationModel = CLIPModel().to(CFG.device)
    sketchClassificationModel.load_state_dict(torch.load(weights_path_sketch + "ClipModel.pt", map_location=CFG.device))
    sketchClassificationModel.eval()
    return sketchClassificationModel

def download_image(image_path):
    bucket = storage.bucket()
    blob = bucket.blob(image_path)
    image_url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
    image_numpy = skimage.io.imread(image_url)
    return image_numpy

def find_top_10(sketch, icon_features, icon_info, session_state):
    sketch = torch.tensor(sketch).permute(0,3,1,2).to(CFG.device, dtype=torch.float)
    sketch_features = sketchClassificationModel.image_encoder(sketch.to(CFG.device))
    sketch_embeddings = sketchClassificationModel.image_projection(sketch_features)
    sketch_embeddings = sketch_embeddings.detach().cpu().numpy()
    # sketch_features = sketchClassificationModel(sketch, training = False)[3]
    sketch_features_tile = np.tile(sketch_embeddings, len(icon_info)).reshape(len(icon_info), 256)
    diff = np.sqrt(np.mean((sketch_features_tile - icon_features)**2, -1))
    top_10 = np.argsort(diff)[:10]
    images_list = []
    for index in top_10:
        image_path = "icon/" + icon_info[index][1] + "/" + icon_info[index][0]
        # img = Image.open(image_path)
        img = download_image(image_path)
        images_list.append(img)
    session_state.top_10_icons = images_list
    st.image(images_list, width=135)

if not firebase_admin._apps:
    firebaseConfig = {
        "type": st.secrets["type"],
        "project_id": st.secrets["project_id"],
        "private_key_id": st.secrets["private_key_id"],
        "private_key": st.secrets["private_key"],
        "client_email": st.secrets["client_email"],
        "client_id": st.secrets["client_id"],
        "auth_uri": st.secrets["auth_uri"],
        "token_uri": st.secrets["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["client_x509_cert_url"]}
    cred = credentials.Certificate(firebaseConfig)
    firebase_admin.initialize_app(cred, {'databaseURL': st.secrets["databaseURL"],'storageBucket': st.secrets["storageBucket"]})

st.title("Sketch-Icon Search Engine")
st.markdown(' ')
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=1,
    stroke_color="rgba(0, 0, 0, 1)",
    background_color="rgba(255, 255, 255, 1)",
    height=448,
    width=448,
    drawing_mode="freedraw",
    key="canvas",
)
result_btn = st.button("Search")

# icon_info = np.load("icon_info_clip.npy")
icon_info = np.load("./Scripts/icon_info_clip.npy", allow_pickle=True)
# icon_features = np.load("icon_features_clip.npy")
icon_features = np.load("./Scripts/icon_features_clip.npy", allow_pickle=True)
sketchClassificationModel = initialize_model()
session_state = SessionState.get(top_10_icons = [])

if result_btn:
    sketch_arr = canvas_result.image_data[:,:,0:3].astype(np.uint8)
    sketch = Image.fromarray(sketch_arr)
    sketch = sketch.resize(size=(224, 224))
    sketch = np.array(sketch)
    sketch = sketch/255
    sketch =  np.expand_dims(sketch, axis=0)
    find_top_10(sketch, icon_features, icon_info, session_state)
else:
    st.image(session_state.top_10_icons, width=135)
