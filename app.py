import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="图像滤波演示", layout="wide")
st.title("🔬 图像滤波演示工具")
st.caption("支持均值滤波、中值滤波、高斯滤波、双边滤波")

FILTER_TYPES = ["均值滤波", "中值滤波", "高斯滤波", "双边滤波"]
DEFAULT_N = [3, 5, 7, 9, 11]
DEFAULT_D = [3, 5, 7, 9, 11]
DEFAULT_SIGMA_COLOR = [50.0, 60.0, 75.0, 100.0, 130.0]
DEFAULT_SIGMA_SPACE = [50.0, 60.0, 75.0, 100.0, 130.0]

uploaded = st.file_uploader("📁 选择一张图片", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # ===== 缩放控制 =====
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

    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    st.markdown("---")

    # ===== 滤波展示 =====
    cols_row1 = st.columns(3)
    cols_row2 = st.columns(3)
    all_cols = cols_row1 + cols_row2

    with all_cols[0]:
        st.markdown("### 🖼️ 原图（缩放后）")
        st.image(img_rgb, use_container_width=True)

    for i in range(5):
        with all_cols[i + 1]:
            st.markdown(f"### 滤波 {i + 1}")

            filter_type = st.selectbox(
                "选择滤波方式",
                FILTER_TYPES,
                index=0,
                key=f"type_{i}"
            )

            if filter_type == "双边滤波":
                d = st.number_input(
                    "邻域直径 d（正奇数）",
                    min_value=1,
                    step=2,
                    value=DEFAULT_D[i],
                    key=f"d_{i}"
                )
                sigma_color = st.number_input(
                    "颜色标准差 (σ_color)",
                    min_value=1.0,
                    max_value=300.0,
                    value=DEFAULT_SIGMA_COLOR[i],
                    step=5.0,
                    key=f"sc_{i}"
                )
                sigma_space = st.number_input(
                    "空间标准差 (σ_space)",
                    min_value=1.0,
                    max_value=300.0,
                    value=DEFAULT_SIGMA_SPACE[i],
                    step=5.0,
                    key=f"ss_{i}"
                )

                if d % 2 == 0:
                    st.error("⚠ d 必须为正奇数！")
                else:
                    d = int(d)
                    st.markdown(
                        f"<p style='font-size:18px; font-weight:bold; color:#FFD700;'>"
                        f"d={d}, σ_color={sigma_color:.1f}, σ_space={sigma_space:.1f}</p>",
                        unsafe_allow_html=True
                    )
                    filtered = cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)
                    st.image(filtered, use_container_width=True)

            else:
                n = st.number_input(
                    "滤波器尺寸 N（正奇数）",
                    min_value=1,
                    step=2,
                    value=DEFAULT_N[i],
                    key=f"n_{i}"
                )

                if n % 2 == 0:
                    st.error("⚠ N 必须为正奇数！")
                else:
                    n = int(n)
                    if filter_type == "均值滤波":
                        st.markdown(
                            f"<p style='font-size:18px; font-weight:bold; color:#FFD700;'>"
                            f"核大小：{n} × {n}</p>",
                            unsafe_allow_html=True
                        )
                        filtered = cv2.blur(img_rgb, (n, n))

                    elif filter_type == "中值滤波":
                        st.markdown(
                            f"<p style='font-size:18px; font-weight:bold; color:#FFD700;'>"
                            f"核大小：{n} × {n}</p>",
                            unsafe_allow_html=True
                        )
                        filtered = cv2.medianBlur(img_rgb, n)

                    elif filter_type == "高斯滤波":
                        sigma = 0.3 * ((n - 1) * 0.5 - 1) + 0.8
                        st.markdown(
                            f"<p style='font-size:18px; font-weight:bold; color:#FFD700;'>"
                            f"标准差 σ = {sigma:.4f}</p>",
                            unsafe_allow_html=True
                        )
                        filtered = cv2.GaussianBlur(img_rgb, (n, n), 0)

                    st.image(filtered, use_container_width=True)

else:
    st.info("👆 请先上传一张图片，然后选择不同的滤波方式进行对比。")
