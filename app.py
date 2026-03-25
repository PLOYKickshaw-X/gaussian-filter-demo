import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="高斯滤波演示", layout="wide")
st.title("🔬 高斯滤波演示工具")

uploaded = st.file_uploader("📁 选择一张图片", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h_orig, w_orig = img_rgb.shape[:2]

    # ========== 新增：图像缩放控制 ==========
    st.markdown("---")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_side = st.slider(
            "📐 缩放后最大边长（像素）",
            min_value=64,
            max_value=min(max(h_orig, w_orig), 2048),
            value=min(512, max(h_orig, w_orig)),
            step=32
        )
    with col_b:
        scale = min(max_side / max(h_orig, w_orig), 1.0)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        st.markdown(
            f"<p style='font-size:18px; margin-top:12px;'>"
            f"原图：<b>{w_orig} × {h_orig}</b> → "
            f"缩放后：<b>{new_w} × {new_h}</b></p>",
            unsafe_allow_html=True
        )

    # 执行缩放
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # ========================================

    cols_row1 = st.columns(3)
    cols_row2 = st.columns(3)
    all_cols = cols_row1 + cols_row2

    with all_cols[0]:
        st.markdown("### 🖼️ 原图（缩放后）")
        st.image(img_rgb, use_container_width=True)

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
