# general package
import streamlit as st
import os
import numpy as np
import time
import cv2
import pickle
import pandas as pd
#radiomics
import radiomics
import SimpleITK as sitk
import six
from radiomics import featureextractor
from radiomics import imageoperations

st.title("BreastUS benign vs malignant classifier test")
st.markdown("<small>by tatsunidas</small>",unsafe_allow_html=True)
st.markdown("[<small>Github</small>](https://github.com/tatsunidas/Radiomics_BreastUS_BvsM)" , unsafe_allow_html=True)
st.markdown("\nThis app predict probability of benign or malignant using breast us imageset (image and mask).")

# button
col, = st.columns(1)
predict_button = col.button('Predict sample data')

@st.cache
def create_model(model_path="model.sav"):
    return pickle.load(open(model_path, 'rb'))


def load_mean_and_var(df_path="scaler_mean_and_variance.csv"):
    return pd.read_csv(df_path, index_col=0)


def get_features(img_path=None, mask_path=None, norm=True):
    settings = {}
    settings['binWidth'] = 25
    # If enabled, resample image (resampled image is automatically cropped.
    settings['resampledPixelSpacing'] = None # [3,3,3] is an example for defining resampling (v
    settings['interpolator'] = sitk.sitkBSpline
    settings['label'] = 1
    settings['force2D'] = True
    # settings['correctMask'] = True #
    extractor = featureextractor.RadiomicsFeatureExtractor(*settings)
    mask = cv2.imread(mask_path) # BGR
    if mask is None:
        raise ValueError('Mask checks failed when image loading...')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask > 0, 1, 0).astype(np.int8)
    mask = np.expand_dims(mask, axis=0)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError('Image loading failed . Can not find image file.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int8)
    img = np.expand_dims(img, axis=0)
    # to sitk image
    img = sitk.GetImageFromArray(img)
    mask = sitk.GetImageFromArray(mask)
    if norm:
        img = imageoperations.normalizeImage(img)
        bb, correctedMask = imageoperations.checkMask(img, mask)
    if correctedMask is not None:
        # Update the mask if it had to be resampled
        mask = correctedMask
    if bb is None: # boundingBox
        raise ValueError('Mask checks failed during pre-crop')
    img, mask = imageoperations.cropToTumorMask(img, mask, bb)
    textures = extractor.computeFeatures(img, mask, "original")
    texture_values = {}
    for key, val in six.iteritems(textures):
        texture_values[key] = [val]
    texture_df = pd.DataFrame(texture_values)
    return texture_df.copy()


def do_predict(loaded_model=None, img_path="", mask_path="", norm=True, scaler_mean_var=None):
    image_1 = cv2.imread(img_path)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY) # .astype(np.int8)
    image_1_view = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)/255
    image_2 = cv2.imread(mask_path)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY) # .astype(np.int8)
    image_2_view = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)/255
    st.image([image_1_view,image_2_view], width=300)
    st.markdown(" ### **Prediction:**")
    if norm:
        if scaler_mean_var is None:
            raise ValueError('Standardization was failed . Please set scale maen and sigma.')
    start = time.process_time()
    f_ = get_features(img_path, mask_path, norm=True)
    f_norm = (f_[:1].values - scaler_mean_var[:1].values) / np.sqrt(scaler_mean_var[1:2].values)
    result = loaded_model.predict(f_norm)
    proba = loaded_model.predict_proba(f_norm)
    basename = os.path.basename(mask_path)
    ans = int(1 if "malignant" in basename else 0)
    title = ""
    if ans == 0:
        # correct or not
        if int(result) == ans:
            title += "ans.benign, success, "
        else:
            title += "ans.benign, error, "
    else:
        if int(result) == ans:
            title += "ans.malignant, success, "
        else:
            title += "ans.malignant, error, "
    title += str(round(proba[0][int(ans)],3)*100)+" %"
    pred_res_string = title
    result = st.empty()
    result.write(pred_res_string)
    time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
    st.write(time_taken)
    del image_1,image_2


def predict_sample(model=None, folder='./test_images', scaler_mean_var=None):
    selected_index = np.random.randint(0, 9)
    sample_images = [
        '/benign (433).png',
        '/benign (434).png',
        '/benign (435).png',
        '/benign (436).png',
        '/benign (437).png',
        '/malignant (206).png',
        '/malignant (207).png',
        '/malignant (208).png',
        '/malignant (209).png',
        '/malignant (210).png'
    ]
    sample_masks = [
        '/benign (433)_mask.png',
        '/benign (434)_mask.png',
        '/benign (435)_mask.png',
        '/benign (436)_mask.png',
        '/benign (437)_mask.png',
        '/malignant (206)_mask.png',
        '/malignant (207)_mask.png',
        '/malignant (208)_mask.png',
        '/malignant (209)_mask.png',
        '/malignant (210)_mask.png'
    ]

    img_path = folder+sample_images[selected_index]
    mask_path = folder+sample_masks[selected_index]
    do_predict(loaded_model=model, img_path=img_path, mask_path=mask_path, norm=True, scaler_mean_var=scaler_mean_var)


scaler_mean_var = load_mean_and_var()
model_ = create_model()

if predict_button:
    predict_sample(model_, scaler_mean_var=scaler_mean_var)

# code end