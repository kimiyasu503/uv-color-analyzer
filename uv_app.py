import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
import plotly.express as px
import plotly.graph_objects as go
from colour.plotting import plot_chromaticity_diagram_CIE1976UCS

# ── 色度図生成→base64 エンコードをキャッシュ化 ────────────────────────
@st.cache_data(show_spinner=False)
def get_encoded_chroma() -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_chromaticity_diagram_CIE1976UCS(show=False, axes=ax)
    ax.set_xlim(-0.01, 0.7)
    ax.set_ylim(-0.01, 0.7)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

encoded_chroma = get_encoded_chroma()

# ── Streamlit セットアップ ─────────────────────────────────────────
st.set_page_config(page_title="JNCD差分分析", layout="wide")
st.sidebar.header("🔧 JNCD色しきい値")

# 動的しきい値スライダー
th1 = st.sidebar.slider("しきい値① (背景 < ①)"    , 0.0, 5.0, 0.2, 0.1)
th2 = st.sidebar.slider("しきい値② (① ≤ 背景 < ②)", 0.0, 5.0, 0.4, 0.1)
th3 = st.sidebar.slider("しきい値③ (② ≤ 背景 < ③)", 0.0, 5.0, 0.8, 0.1)
th4 = st.sidebar.slider("しきい値④ (③ ≤ 背景 < ④)", 0.0, 5.0, 1.2, 0.1)
th5 = st.sidebar.slider("しきい値⑤ (④ ≤ 背景 < ⑤)", 0.0, 5.0, 1.6, 0.1)

st.title("🔍 比較データ差分分析（u′v′ & JNCD）")

# ── ファイルアップロード ─────────────────────────────────────────
up1 = st.file_uploader("📂 比較データ①", type="txt", key="file1")
up2 = st.file_uploader("📂 比較データ②", type="txt", key="file2")

# ── パーサ定義 ─────────────────────────────────────────────────
def parse_gr55_xyz(content: str) -> pd.DataFrame:
    recs = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("sampleid"):
            continue
        tokens = [t for t in re.split(r"[,\t ]+", line) if t]
        if len(tokens) < 4:
            continue
        try:
            sid = int(tokens[0])
            x, y, z = map(float, tokens[-3:])
            recs.append((sid, x, y, z))
        except ValueError:
            continue
    return pd.DataFrame(recs, columns=["SampleID", "X", "Y", "Z"])

def xyz_to_uv(X, Y, Z):
    denom = X + 15 * Y + 3 * Z
    if denom == 0 or np.isnan(denom):
        return 0.0, 0.0
    return 4 * X / denom, 9 * Y / denom

# ── メイン処理 ─────────────────────────────────────────────────
if up1 and up2:
    # ファイル読み込み & パース
    c1 = up1.getvalue().decode("utf-8", errors="ignore")
    c2 = up2.getvalue().decode("utf-8", errors="ignore")
    df1 = parse_gr55_xyz(c1)
    df2 = parse_gr55_xyz(c2)

    # 連番インデックスを除去し、ID列をインデックスに
    df1_disp = df1.rename(columns={"SampleID": "ID"}).set_index("ID")
    df2_disp = df2.rename(columns={"SampleID": "ID"}).set_index("ID")

    # 元データ表示
    st.subheader(f"比較データ①（件数: {len(df1_disp)}）")
    st.dataframe(df1_disp)
    st.subheader(f"比較データ②（件数: {len(df2_disp)}）")
    st.dataframe(df2_disp)

    if df1.empty or df2.empty:
        st.error("⚠️ データが抽出できません。ファイルを確認してください。")
        st.stop()

    # 共通ID抽出＋u′v′変換
    common = set(df1.SampleID) & set(df2.SampleID)
    if not common:
        st.error("⚠️ 共通 SampleID がありません。")
        st.stop()

    df1 = df1[df1.SampleID.isin(common)].sort_values("SampleID")
    df2 = df2[df2.SampleID.isin(common)].sort_values("SampleID")
    df1[["u1", "v1"]] = df1.apply(lambda r: pd.Series(xyz_to_uv(r.X, r.Y, r.Z)), axis=1)
    df2[["u2", "v2"]] = df2.apply(lambda r: pd.Series(xyz_to_uv(r.X, r.Y, r.Z)), axis=1)

    # マージ & JNCD算出
    merged = pd.merge(df1, df2, on="SampleID")
    merged["JNCD"] = np.sqrt((merged.u2 - merged.u1)**2 + (merged.v2 - merged.v1)**2) / 0.004

    # 🎨 JNCD 色分け基準 (動的設定) — カラースウォッチ表示
    st.markdown("### 🎨 JNCD 色分け基準 (動的設定)")
    swatches = [
        (f"JNCD < {th1:.1f}",           "#FFFFFF"),   # 白
        (f"{th1:.1f} ≤ JNCD < {th2:.1f}", "#66CC33"),  # 濃緑
        (f"{th2:.1f} ≤ JNCD < {th3:.1f}", "#3399CC"),  # 濃青
        (f"{th3:.1f} ≤ JNCD < {th4:.1f}", "#FFCC00"),  # 濃黄
        (f"{th4:.1f} ≤ JNCD < {th5:.1f}", "#9933CC"),  # 濃紫
        (f"JNCD ≥ {th5:.1f}",           "#CC0033"),   # 濃赤
    ]
    for label, color in swatches:
        st.markdown(
            f'<span style="display:inline-block;width:16px;height:16px;'
            f'background-color:{color};border:1px solid #444;'
            f'margin-right:6px;vertical-align:middle;"></span>{label}',
            unsafe_allow_html=True
        )

    # セル背景色関数（読みやすい濃度に）
    def cell_bg(j):
        if j < th1:
            return "background-color: #FFFFFF"
        elif j < th2:
            return "background-color: #66CC33"
        elif j < th3:
            return "background-color: #3399CC"
        elif j < th4:
            return "background-color: #FFCC00"
        elif j < th5:
            return "background-color: #9933CC"
        else:
            return "background-color: #CC0033"

    # JNCD 一覧テーブル（インデックス非表示・ID左寄せ・色分け）
    tbl = (
        merged[["SampleID", "X_x", "Y_x", "Z_x", "u1", "v1", "X_y", "Y_y", "Z_y", "u2", "v2", "JNCD"]]
        .rename(columns={"SampleID": "ID", "X_x": "X①", "Y_x": "Y①", "Z_x": "Z①", "X_y": "X②", "Y_y": "Y②", "Z_y": "Z②"})
    )
    def jncd_excel_bg(val):
        color = cell_bg(val).replace('background-color: ', '')
        return f'background-color: {color}'
    styled = (
        tbl.style
           .applymap(jncd_excel_bg, subset=["JNCD"])
           .set_properties(subset=["ID"], **{"text-align": "left"})
           .hide(axis="index")
    )
    st.subheader("📋 JNCD 一覧（色分け）")
    st.write(styled)

    # JNCD ヒストグラム
    st.subheader("📊 JNCD ヒストグラム")
    # JNCD値ごとに色分け
    def jncd_hist_color(j):
        if j >= th5:
            return "#CC0033"
        if j >= th4:
            return "#9933CC"
        if j >= th3:
            return "#FFCC00"
        if j >= th2:
            return "#3399CC"
        if j >= th1:
            return "#66CC33"
        return "#FFFFFF"

    merged["HistColor"] = merged["JNCD"].apply(jncd_hist_color)
    fig_hist = go.Figure()
    bins = np.arange(0, merged["JNCD"].max()+0.1, 0.1)
    hist, edges = np.histogram(merged["JNCD"], bins=bins)
    for i in range(len(hist)):
        fig_hist.add_trace(go.Bar(
            x=[edges[i]],
            y=[hist[i]],
            marker=dict(color=jncd_hist_color(edges[i]), line=dict(color="black", width=1)),
            width=0.09,
            name=f"{edges[i]:.1f}～{edges[i+1]:.1f}"
        ))
    fig_hist.update_layout(
        title="JNCD の分布",
        xaxis_title="JNCD 値",
        yaxis_title="サンプル数",
        xaxis=dict(dtick=0.1),
        bargap=0.1,
        showlegend=False,
        width=700,
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # u′v′プロット（JNCD色分け + 色度図背景）
    st.subheader("🖼️ u′v′プロット（JNCD色分け）")
    fig_j = go.Figure()
    fig_j.add_layout_image(dict(
        source="data:image/png;base64," + encoded_chroma,
        xref="x", yref="y",
        x=0.0, y=0.7, sizex=0.7, sizey=0.7,
        sizing="stretch", opacity=0.6, layer="below"
    ))
    merged["Color"] = merged.JNCD.apply(lambda j: cell_bg(j).split(": ")[1])
    fig_j.add_trace(go.Scatter(
        x=merged.u1, y=merged.v1, mode="markers",
        marker=dict(color=merged.Color, size=12, line=dict(color="white", width=1)),
        name="比較①（色分け）",
        text=[f"ID {sid}<br>u1: {u:.4f}<br>v1: {v:.4f}<br>JNCD: {j:.2f}" for sid, u, v, j in zip(merged.SampleID, merged.u1, merged.v1, merged.JNCD)],
        hoverinfo="text"
    ))
    fig_j.add_trace(go.Scatter(
        x=merged.u2, y=merged.v2, mode="markers",
        marker=dict(color="black", size=4, symbol="cross", opacity=1.0),
        name="比較②（＋）",
        text=merged.SampleID,
        hovertemplate="u2:%{x:.4f}<br>v2:%{y:.4f}<br>ID %{text}"
    ))
    fig_j.update_layout(
        width=900,
        height=900,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            title="u′",
            range=[-0.01, 0.7],
            fixedrange=True,
            scaleanchor="y",
            scaleratio=1,
            domain=[0.0, 1.0]
        ),
        yaxis=dict(
            title="v′",
            range=[-0.01, 0.7],
            fixedrange=True,
            domain=[0.0, 1.0]
        ),
        showlegend=False,
        annotations=[
            dict(
                x=0.98, y=0.98, xref="paper", yref="paper",
                text="<span style='color:black;font-size:18px;'>○ </span> 比較①（色分け）<br><span style='color:black;font-size:18px;'>＋</span> 比較②",
                showarrow=False,
                align="left",
                font=dict(size=16),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="gray",
                borderwidth=1
            )
        ]
    )
    st.plotly_chart(fig_j, use_container_width=False)

    # Excelダウンロード（画像・元データ付）
    buf2 = io.BytesIO()
    # 画像保存（ヒストグラム）
    hist_img = io.BytesIO()
    fig_hist.write_image(hist_img, format="png")
    hist_img.seek(0)
    # 画像保存（u′v′プロット）
    plot_img = io.BytesIO()
    fig_j.write_image(plot_img, format="png")
    plot_img.seek(0)

    with pd.ExcelWriter(buf2, engine="xlsxwriter") as writer:
        # オートフィルタ範囲取得
        col_count = len(tbl.columns)
        # 共通変数定義
        workbook  = writer.book
        import datetime
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fname1 = up1.name if up1 is not None and hasattr(up1, 'name') else '比較データ①'
        fname2 = up2.name if up2 is not None and hasattr(up2, 'name') else '比較データ②'

        # JNCD結果（元順）
        ws = workbook.add_worksheet("JNCD結果")
        ws.write(0, 0, f"比較データ①ファイル: {fname1}")
        ws.write(1, 0, f"比較データ②ファイル: {fname2}")
        ws.write(2, 0, f"出力日時: {ts}")
        for col_idx, col_name in enumerate(tbl.columns):
            ws.write(3, col_idx, col_name)
        ws.autofilter(3, 0, 3+len(tbl), col_count-1)
        for row_idx, row in enumerate(tbl.values):
            for col_idx, val in enumerate(row):
                if tbl.columns[col_idx] == "JNCD":
                    color = cell_bg(val).replace('background-color: ', '')
                    ws.write(row_idx+4, col_idx, val, workbook.add_format({'bg_color': color}))
                else:
                    ws.write(row_idx+4, col_idx, val)
            ws.insert_image("N6", "hist.png", {'image_data': hist_img})
            ws.insert_image("N30", "plot.png", {'image_data': plot_img})

        # JNCD結果（ID昇順）
        tbl_id = tbl.sort_values("ID")
        ws_id = workbook.add_worksheet("JNCD結果（ID昇順）")
        ws_id.write(0, 0, f"比較データ①ファイル: {fname1}")
        ws_id.write(1, 0, f"比較データ②ファイル: {fname2}")
        ws_id.write(2, 0, f"出力日時: {ts}")
        for col_idx, col_name in enumerate(tbl_id.columns):
            ws_id.write(3, col_idx, col_name)
        ws_id.autofilter(3, 0, 3+len(tbl_id), col_count-1)
        for row_idx, row in enumerate(tbl_id.values):
            for col_idx, val in enumerate(row):
                if tbl_id.columns[col_idx] == "JNCD":
                    color = cell_bg(val).replace('background-color: ', '')
                    ws_id.write(row_idx+4, col_idx, val, workbook.add_format({'bg_color': color}))
                else:
                    ws_id.write(row_idx+4, col_idx, val)
            ws_id.insert_image("N6", "hist.png", {'image_data': hist_img})
            ws_id.insert_image("N30", "plot.png", {'image_data': plot_img})

        # JNCD結果（JNCD降順）
        tbl_jncd = tbl.sort_values("JNCD", ascending=False)
        ws_jncd = workbook.add_worksheet("JNCD結果（JNCD降順）")
        ws_jncd.write(0, 0, f"比較データ①ファイル: {fname1}")
        ws_jncd.write(1, 0, f"比較データ②ファイル: {fname2}")
        ws_jncd.write(2, 0, f"出力日時: {ts}")
        for col_idx, col_name in enumerate(tbl_jncd.columns):
            ws_jncd.write(3, col_idx, col_name)
        ws_jncd.autofilter(3, 0, 3+len(tbl_jncd), col_count-1)
        for row_idx, row in enumerate(tbl_jncd.values):
            for col_idx, val in enumerate(row):
                if tbl_jncd.columns[col_idx] == "JNCD":
                    color = cell_bg(val).replace('background-color: ', '')
                    ws_jncd.write(row_idx+4, col_idx, val, workbook.add_format({'bg_color': color}))
                else:
                    ws_jncd.write(row_idx+4, col_idx, val)
            ws_jncd.insert_image("N6", "hist.png", {'image_data': hist_img})
            ws_jncd.insert_image("N30", "plot.png", {'image_data': plot_img})
    buf2.seek(0)
    st.download_button(
        label="📤 Excelに保存（画像・元データ付）",
        data=buf2,
        file_name="jncd_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
# ── ここまで ────────────────────────────────────────────────