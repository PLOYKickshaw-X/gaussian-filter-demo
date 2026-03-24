import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="高斯滤波演示", layout="wide")
st.title("🔬 高斯滤波演示工具")

# 上传图片
uploaded = st.file_uploader("📁 选择一张图片", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 两行三列
    cols_row1 = st.columns(3)
    cols_row2 = st.columns(3)
    all_cols = cols_row1 + cols_row2

    # 第一个格子：原图
    with all_cols[0]:
        st.markdown("### 🖼️ 原图")
        st.image(img_rgb, use_container_width=True)

    # 第2~6个格子：滤波
    for i in range(1, 6):
        with all_cols[i]:
            st.markdown(f"### 滤波 {i}")
            default_n = i * 2 + 1
            n = st.number_input(
                f"滤波器大小 N（正奇数）",
                min_value=1,
                step=2,
                value=default_n,
                key=f"n_{i}"
            )

            if n % 2 == 0:
                st.error("⚠ N 必须为正奇数！")
            else:
                sigma = 0.3 * ((n - 1) * 0.5 - 1) + 0.8
                st.markdown(
                    f"<p style='font-size:22px; font-weight:bold; color:#FFD700;'>"
                    f"σ = {sigma:.4f}</p>",
                    unsafe_allow_html=True
                )
                filtered = cv2.GaussianBlur(img_rgb, (int(n), int(n)), 0)
                st.image(filtered, use_container_width=True)
else:
    st.info("👆 请先上传一张图片，然后体验不同 N 值的高斯滤波效果。")
