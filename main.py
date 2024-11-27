import time
import os
import numpy as np
import streamlit as st

from convolve import create_line_psf
from deblur import computeLocalPrior, updatePsi, computeL, updatef, save_mask_as_image
from helpers import open_image, write_image, kernel_from_image

st.header("盲卷积图像去模糊算法")
st.subheader("Authored by: 杨竣杰")

# 设置最大迭代次数
MAX_ITER = st.slider("Choose max iteration", 0, 30, value=10)

# 关键变量
VARS = {
    'gamma': 2,  # 初始值，随后会倍增
    'lambda1': 0.5,  # 范围 [0.002, 0.5]
    'k1': 1.1,  # 范围 [1.1, 1.4]
    'lambda2': 25,  # 范围 [10, 25]
    'k2': 1.5,
}


# 图像路径字典
images_dict = {
    "picasso": {
        "blurred": "blur_pic/picassoBlurImage.png",
        "target": "blur_pic/picassoOut.png"
    },
    "nv": {
        "blurred": "blur_pic/nvBlurImage.png",
        "target": "blur_pic/nvOut.png"
    },
    "glass ball": {
        "blurred": "blur_pic/glassBallBlur1.png",
        "target": "blur_pic/glassBall.png"
    },
    "landscape": {
        "blurred": "blur_pic/landscapeBlur1.png",
        "target": "blur_pic/landscape.png"
    },
    "squirrel": {
        "blurred": "blur_pic/squirrelBlur1.png",
        "target": "blur_pic/squirrel.png"
    },
}

# 状态管理初始化
if "running" not in st.session_state:
    st.session_state["running"] = False

# UI：选择图像对
selected_image_pair = st.selectbox(
    "Select an image pair for deblurring:",
    list(images_dict.keys())
)

if st.button("Finish select", type="primary", use_container_width=True):
    # 保存选定的图像对
    st.session_state["selected_pair"] = selected_image_pair
    st.success(f"Selected {selected_image_pair} for processing.")

# 检查是否选择了图像对
if "selected_pair" not in st.session_state:
    st.warning("Please select an image pair to proceed.")
    st.stop()

# 获取选定图像路径
selected_pair = st.session_state["selected_pair"]
blurred_path = images_dict[selected_pair]["blurred"]
target_path = images_dict[selected_pair]["target"]

# 使用 st.columns 创建两列布局
col1, col2 = st.columns(2)

# 在第一列显示模糊图像
with col1:
    st.image(blurred_path, caption="Blurred Image", use_container_width=True)

# 在第二列显示目标图像
with col2:
    st.image(target_path, caption="Target Image", use_container_width=True)


# 按钮事件触发
if st.button("Start running!", type="primary", use_container_width=True):
    st.session_state["running"] = True

if st.session_state["running"]:
    try:
        # 加载图像数据
        I = np.atleast_3d(open_image(blurred_path))
        target_image = np.atleast_3d(open_image(target_path))

        st.success("Picture loaded.")

        # 初始化变量
        f = create_line_psf(-np.pi / 4, 1, (27, 27))  # 初始卷积核
        n_rows = 260
        L = I.copy()
        nL = I.copy()

        # 计算 Omega 区域
        O_THRESHOLD = 5
        M = np.zeros_like(I)
        for i in range(I.shape[2]):
            M[:, :, i] = computeLocalPrior(I[:, :, i], f.shape, O_THRESHOLD)

        st.success("Compute Omega zone.")

        # 计算图像梯度
        I_d = [np.gradient(I[:, :, i], axis=(1, 0)) for i in range(I.shape[2])]

        # 初始化 Psi
        Psi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]
        nPsi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]

        st.success("Compute gradient and deblur kernel.")

        iterations = 0
        deblur_placeholder = st.empty()
        kernel_placeholder = st.empty()
        progress_bar = st.progress(0)

        # Δ值可视化
        delta_chart_placeholder = st.empty()
        delta_values = []

        while iterations < MAX_ITER:
            VARS['gamma'] = 2
            delta = 5000
            iters = 0

            while iters < 1:
                for i in range(L.shape[2]):
                    L_d = np.gradient(L[:, :, i], axis=(1, 0))
                    nPsi[i] = updatePsi(I_d[i], L_d, M[:, :, i], VARS['lambda1'], VARS['lambda2'], VARS['gamma'])
                    nL[:, :, i] = computeL(L[:, :, i], I[:, :, i], f, nPsi[i], VARS['gamma'])
                deltaL = nL - L
                delta = np.linalg.norm(deltaL)
                delta_values.append(delta)
                delta_chart_placeholder.line_chart(delta_values)
                L = nL.copy()
                nPsi = Psi.copy()
                VARS['gamma'] *= 2
                iters += 1

            # 保存当前迭代结果
            deblur_image_path = f'picture\deblurred_iteration_{iterations}.png'
            kernel_image_path = f'picture\kernel_iteration_{iterations}.png'
            os.makedirs(f'picture', exist_ok=True)
            write_image(deblur_image_path, L.copy())
            write_image(kernel_image_path, f.copy() * (255 / np.max(f)))

            # 更新显示
            deblur_placeholder.image(deblur_image_path, caption=f"Deblurred Image (Iteration {iterations})")
            kernel_placeholder.image(kernel_image_path, caption=f"Kernel (Iteration {iterations})", width=250)

            progress_bar.progress((iterations + 1) / MAX_ITER)

            # 更新卷积核
            f = updatef(L, I, f, n_rows=n_rows, k_cut_ratio=0)
            VARS['lambda1'] /= VARS['k1']
            VARS['lambda2'] /= VARS['k2']
            iterations += 1

        # 计算最终 MSE
        mse = np.mean((L - target_image) ** 2)
        st.write(f"Final MSE against the target image: {mse:.4f}")
        st.success("All iterations completed!")
        
        # 计算模糊图像和目标图像之间的 MSE
        blurred_mse = np.mean((I - target_image) ** 2)

        # 计算提升百分比
        improvement_percentage = ((blurred_mse - mse) / blurred_mse) * 100

        # 输出结果
        st.write(f"MSE between blurred image and target image: {blurred_mse:.4f}")
        st.write(f"Improvement percentage: {improvement_percentage:.2f}%")


    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        st.session_state["running"] = False
