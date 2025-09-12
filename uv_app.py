# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import io
import os

st.set_page_config(page_title="UV座標変換 & JNCD分析ツール", layout="wide")
st.title("🔬 UV座標変換 & JNCD分析ツール")

# ファイルアップロード
uploaded_file = st.file_uploader("QLED.TXTを選択してください", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8", errors="ignore")

    # 柔軟なXYZ抽出関数
    def parse_xyz_lines(content):
        xyz_list = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [t for t in line.replace(",", " ").replace("\t", " ").split() if t]
            if len(tokens) >= 3:
                try:
                    x, y, z = map(float, tokens[:3])
                    xyz_list.append((x, y, z))
                except ValueError:
                    continue
        return pd.DataFrame(xyz_list, columns=["X", "Y", "Z"])

    df = parse_xyz_lines(content)

    if df.empty:
        st.error("XYZデータが見つかりません。ファイル形式をご確認ください。")
    else:
        # XYZ → u’v’変換（CIE 1976）
        def xyz_to_uv(X, Y, Z):
            denom = X + 15 * Y + 3 * Z
            if denom == 0: return (0, 0)
            u = 4 * X / denom
            v = 9 * Y / denom
            return (u, v)

        df[['u', 'v']] = df.apply(lambda row: pd.Series(xyz_to_uv(row['X'], row['Y'], row['Z'])), axis=1)

        st.subheader("📋 u’v’座標一覧")
        st.dataframe(df)

        # Δuv（JNCD）マトリクス計算
        n = len(df)
        jncd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                u1, v1 = df.loc[i, ['u', 'v']]
                u2, v2 = df.loc[j, ['u', 'v']]
                jncd_matrix[i, j] = np.sqrt((u1 - u2)**2 + (v1 - v2)**2) / 0.004

        # インタラクティブヒートマップ（Plotly）
        st.subheader("🖱️ JNCDヒートマップ（マウスオーバー対応）")
        fig = go.Figure(data=go.Heatmap(
            z=jncd_matrix,
            x=list(range(n)),
            y=list(range(n)),
            colorscale='RdBu',
            reversescale=True,
            hovertemplate='Sample %{x} vs %{y}<br>JNCD: %{z:.2f}<extra></extra>'
        ))
        fig.update_layout(width=800, height=700)
        st.plotly_chart(fig, use_container_width=True)

        # ヒストグラム表示
        st.subheader("📊 JNCD値の分布（ヒストグラム）")
        jncd_values = [jncd_matrix[i][j] for i in range(n) for j in range(i+1, n)]
        hist_fig = px.histogram(
            x=jncd_values,
            nbins=30,
            labels={'x': 'JNCD値'},
            title='色差の分布'
        )
        hist_fig.update_layout(xaxis_title='JNCD', yaxis_title='ペア数', bargap=0.1)
        st.plotly_chart(hist_fig, use_container_width=True)

        # 閾値フィルタ
        st.subheader("🔍 JNCDが閾値以下のペア抽出")
        threshold = st.slider("閾値を選択（例：1.0）", 0.1, 10.0, 1.0, 0.1)
        pairs = [(i, j, jncd_matrix[i, j]) for i in range(n) for j in range(i+1, n) if jncd_matrix[i, j] < threshold]
        st.write(pd.DataFrame(pairs, columns=["Sample A", "Sample B", "JNCD"]))

        # Excel出力
        if st.button("📤 Excelに保存"):
            out_path = os.path.join("B:/Aphrodi/UVConverter", "uv_results.xlsx")
            with pd.ExcelWriter(out_path) as writer:
                df.to_excel(writer, sheet_name="uv_coordinates", index=False)
                pd.DataFrame(jncd_matrix).to_excel(writer, sheet_name="jncd_matrix", index=False)
                pd.DataFrame(pairs, columns=["Sample A", "Sample B", "JNCD"]).to_excel(writer, sheet_name="jncd_filtered", index=False)
            st.success(f"保存完了：{out_path}")
