from urllib.parse import urlparse

# =========================
# URLチェック
# =========================

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc != ""
    
import plotly.graph_objects as go
import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup

# =========================
# モデル読み込み
# =========================

model = joblib.load("classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# =========================
# 画面設定
# =========================

st.set_page_config(
    page_title="Misinformation Analysis System",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>

/* タイトル */
h1 {
    text-align: center;
    color: #4F8BF9;
    font-weight: 700;
}

/* metricカード */
div[data-testid="metric-container"] {
    background-color: #1c2333;
    border: 1px solid #4F8BF9;
    padding: 15px;
    border-radius: 15px;
}

/* ボタン */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

/* プログレスバー */
.stProgress > div > div > div > div {
    background-color: #4F8BF9;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>Misinformation Analysis System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
    text-align:center;
    color:#94A3B8;
    font-size:18px;
    margin-top:-10px;
    margin-bottom:20px;
    ">
    AIによる誤情報拡散パターン分析システム
    </div>
    """,
    unsafe_allow_html=True
)

url = ""

# =========================
# URL入力
# =========================

url = st.text_input(
        "ニュース記事URLを入力してください"
)

if url:

    if not is_valid_url(url):
        st.error("URL形式が正しくありません。")
        st.stop()

if url:

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
            "script",
            "style",
            "iframe",
            "noscript"
            ]
        ):
            tag.decompose()
    
        text = soup.get_text(
            " ",
            strip=True
        )

if len(text) < 300:

    st.error(
        "記事本文が短いため分析できません。"
    )
    st.stop()

        st.success(
            "記事を取得しました"
        )

        with st.expander(
            "取得テキスト確認"
        ):

            st.text_area(
                "先頭1000文字",
                text[:1000],
                height=200
            )

    except Exception as e:

        st.error(
            f"記事取得に失敗しました: {e}"
        )
                                                                                              
# =========================
# 分析開始
# =========================

if st.button(
    "分析開始",
    type="primary"
):

    if text.strip() == "":

        st.warning(
            "文章またはURLを入力してください"
        )

    else:

        with st.spinner(
            "AIが分析中です..."
        ):

            x = vectorizer.transform(
                [text]
            )

            prob = model.predict_proba(
                x
            )[0]

            fake_prob = prob[0] * 100
            true_prob = prob[1] * 100

        # =====================
        # 判定
        # =====================

        if fake_prob >= 75:

            result = "False"
            result_color = "#EF4444"
        
        elif fake_prob >= 50:
        
            result = "Caution"
            result_color = "#F59E0B"
        
        else:
        
            result = "True"
            result_color = "#22C55E"
    
        # =====================
        # 判定結果
        # =====================

        st.markdown(f"""
        <div style="
        background:linear-gradient(
        145deg,
        #1e293b,
        #0f172a
        );
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:48px;
        font-weight:bold;
        color:{result_color};
        text-shadow:0 0 15px {result_color};
        ">
        {result}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # =====================
        # AI信頼度メーター
        # =====================
        
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=true_prob,
                title={
                    "text": "判定信頼度"
                },
                number={
                    "suffix": "%"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "#60A5FA",
                        "thickness": 0.3
                    },
                    "bgcolor": "#0f172a",
                    "borderwidth": 2,
                    "bordercolor": "#334155",
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "#1e293b"
                        },
                        {
                            "range": [50, 75],
                            "color": "#334155"
                        },
                        {
                            "range": [75, 100],
                            "color": "#475569"
                        }
                    ]
                }
            )
        )
        
        fig.update_layout(
            height=320,
            paper_bgcolor="#111827",
            font={
                "color": "white"
            },
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================
        # 分析サマリー
        # =====================

        st.markdown(
            "### 分析サマリー"
        )

        col3, col4 = st.columns(2)

        with col3:

            st.metric(
                "文字数",
                len(text)
            )

        with col4:

            st.metric(
                "特徴語数",
                x.nnz
            )

        # =====================
        # 特徴語
        # =====================

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        scores = x.toarray()[0]

        top_indices = (
            scores.argsort()[-20:][::-1]
        )

        important_words = []

        for idx in top_indices:

            if scores[idx] > 0:

                word = feature_names[idx]

                if (
                    len(word) >= 3
                    and not word.isdigit()
                ):
                    important_words.append(
                        word
                    )

        st.markdown(
            "### 検出特徴語"
        )

        if important_words:

            for word in important_words:

                st.write(
                    f"• {word}"
                )

        else:

            st.write(
                "特徴語が検出されませんでした"
            )

# =========================
# フッター
# =========================

st.markdown("---")

st.caption(
    "本システムは研究・学習目的で開発されています。判定結果は参考情報です。"
)
