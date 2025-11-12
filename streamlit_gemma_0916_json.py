import streamlit as st
import pandas as pd
import io
from datetime import datetime
import isodate
import re
from googleapiclient.discovery import build
import ollama
import subprocess
import time
import requests
import os
import glob
import json

# --- Ollamaの確認 ---
def check_ollama_status():
    """Ollamaの状態をチェックする"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200
    except:
        return False

def start_ollama_service():
    """Ollamaサービスを起動する（バックグラウンド）"""
    try:
        # Windowsの場合
        if os.name == 'nt':
            subprocess.Popen(['ollama', 'serve'], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        # Mac/Linuxの場合
        else:
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        time.sleep(3)  # サービス起動待ち
        return True
    except Exception as e:
        st.error(f"Ollama起動エラー: {e}")
        return False

def ensure_ollama_running():
    """Ollamaが実行中であることを確認し、必要に応じて起動"""
    if not check_ollama_status():
        st.info("Ollamaサービスを起動しています...")
        if start_ollama_service():
            # 最大10秒待機
            for i in range(10):
                if check_ollama_status():
                    st.success("Ollamaサービスが起動しました")
                    return True
                time.sleep(1)
        return False
    return True

def check_model_exists(model_name):
    """モデルが存在するかチェック"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        return model_name.split(':')[0] in result.stdout
    except:
        return False

def pull_model_if_needed(model_name):
    """必要に応じてモデルをプル"""
    try:
        if not check_model_exists(model_name):
            st.info(f"モデル {model_name} をダウンロード中...")
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=300)  # 5分のタイムアウト
            if result.returncode == 0:
                st.success(f"モデル {model_name} のダウンロード完了")
                return True
            else:
                st.error(f"モデルダウンロードエラー: {result.stderr}")
                return False
        return True
    except subprocess.TimeoutExpired:
        st.error("モデルダウンロードがタイムアウトしました")
        return False
    except Exception as e:
        st.error(f"モデル確認エラー: {e}")
        return False

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube + Gemma3 動画構成案ジェネレータ", layout="centered")
st.title("YouTube + Gemma3 動画構成案ジェネレータ")

# --- モデル切替UI（4b / 12b） ---
MODEL_OPTIONS = ("gemma3:4b", "gemma3:12b")

# 既定は12b（必要なら4bに変更OK）
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "gemma3:4b"

with st.sidebar:
    st.subheader("Ollama モデル切替")
    st.session_state["ollama_model"] = st.radio(
        "Gemma3 サイズ",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state["ollama_model"]),
        horizontal=True
    )
    st.caption(f"現在のモデル: **{st.session_state['ollama_model']}**")

def current_model() -> str:
    """選択中モデルを返す（ollama.chat で使う）"""
    return st.session_state["ollama_model"]

CACHE_DIR = ""  # CSV を保存しているフォルダ

# --- YouTube API Key 設定 ---
API_KEY = "AIzaSyC36r9O-Dx4-afYBS1Fpuf_P1K9wpVfsVo"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# --- アプリ起動時のOllama初期化 ---
if 'ollama_initialized' not in st.session_state:
    with st.spinner("Ollama初期化中..."):
        # Ollamaサービスの確認と起動
        if ensure_ollama_running():
            # モデルの確認とダウンロード
            if pull_model_if_needed(current_model()):
                st.session_state['ollama_initialized'] = True
                st.success("初期化完了")
            else:
                st.error("モデルの準備に失敗しました")
                st.stop()
        else:
            st.error("Ollamaサービスの起動に失敗しました")
            st.info("ターミナルで 'ollama serve' を実行してください")
            st.stop()

# --- Ollama状態表示（サイドバー） ---
st.sidebar.header("Ollama状態")
ollama_status = check_ollama_status()
status_color = "🟢" if ollama_status else "🔴"
st.sidebar.write(f"{status_color} Ollama: {'稼働中' if ollama_status else '停止中'}")

# 手動リフレッシュボタン
if st.sidebar.button("状態を更新"):
    st.rerun()

# --- Functions ---
def fetch_youtube_data(query, max_results=200):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    video_data = []
    next_page_token = None

    def change_iso(duration_str):
        try:
            td = isodate.parse_duration(duration_str)
            return str(td)
        except Exception:
            return "0:00"

    while len(video_data) < max_results:
        remaining = max_results - len(video_data)
        search_response = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=min(50, remaining),
            type="video",
            pageToken=next_page_token
        ).execute()

        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
        if not video_ids:
            break

        videos_response = youtube.videos().list(
            id=','.join(video_ids),
            part="snippet,statistics,contentDetails"
        ).execute()

        for item in videos_response.get("items", []):
            video_data.append({
                "videoId": item["id"],
                "title": item["snippet"]["title"],
                "viewCount": int(item["statistics"].get("viewCount", 0)),
                "likeCount": int(item["statistics"].get("likeCount", 0)),
                "duration": change_iso(item["contentDetails"]["duration"]),
                "description": item["snippet"].get("description", "")
            })

        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(video_data)

def generate_suggestions(df, query, num=5, save_path=None):
    if save_path:
        save_file = save_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"suggestions_{ts}.csv"

    # データの要約を作成（最初の10件のタイトルのみ使用）
    sample_titles = df.head(10)['title'].tolist()
    titles_text = "\n".join(sample_titles)
    
    prompt = f"""
あなたは優秀なコピーライターです。
検索キーワード「{query}」で見つかった動画タイトルの例：

{titles_text}

これらの動画の共通テーマから、関連する新しいトピックを{num}個生成してください。
地名・製品名・人物名などの固有名は避け、一般的なトピックにしてください。

以下のCSV形式で出力してください：
```csv
keyword
トピック1
トピック2
...
```
"""

    try:
        response = ollama.chat(
            model=current_model(),
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7, 'num_predict': 500}
        )
        text = response['message']['content']
    except Exception as e:
        st.error(f"キーワード生成エラー: {e}")
        return pd.DataFrame()
    
    # CSV部分を抽出
    m = re.search(r"```csv\n(.*?)```", text, re.DOTALL)
    if m:
        csv_content = m.group(1).strip()
    else:
        # ```csv がない場合、keyword で始まる行を探す
        lines = text.split('\n')
        csv_lines = []
        in_csv = False
        for line in lines:
            if 'keyword' in line.lower():
                in_csv = True
            if in_csv and line.strip():
                csv_lines.append(line.strip())
        csv_content = '\n'.join(csv_lines)

    try:
        sug_df = pd.read_csv(io.StringIO(csv_content))
        # ヘッダーを normalize
        sug_df.columns = [c.strip().lower().replace("キーワード", "keyword") for c in sug_df.columns]

        if 'keyword' not in sug_df.columns:
            st.error("CSVに 'keyword' 列が含まれていません。")
            return pd.DataFrame()

        sug_df = sug_df[['keyword']]
        sug_df.to_csv(save_file, index=False)
        st.success(f"提案キーワードを保存: {save_file}")

        return sug_df

    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return pd.DataFrame()

def compose_structure(base_kw, suggestion, target="特になし", duration=15, purpose="特になし", save_dir="outlines", sections=4):
    full_topic = f"{base_kw} {suggestion}"

    conditions = [f"- {sections}つのセクションに分ける"]
    conditions.append(f"- 動画の長さ: {duration}分")
    if target and target != "特になし":
        conditions.append(f"- 対象視聴者: {target}")
    if purpose and purpose != "特になし":
        conditions.append(f"- 動画の目的: {purpose}")
    conditions += ["- 初心者にも分かりやすい構成"]

    prompt = f"""
        YouTubeの解説の概要文とハッシュタグ・キーワード，サムネイル画像に入れる文言，動画のアウトラインを作成してください。

        テーマ: 「{full_topic}」

        条件:
        {chr(10).join(conditions)}

        以下のキーを持つJSON形式で出力してください。
        - "title": "動画のタイトル案"
        - "summary": "動画の概要文（説明欄用）"
        - "hashtags": ["ハッシュタグ1", "ハッシュタグ2", ...]
        - "keywords": ["関連キーワード1", "関連キーワード2", ...]
        - "thumbnail_text": "サムネイル画像に入れるとクリックされやすい文言"
        - "outline": [
            {{
                "section_title": "セクション1のタイトル+時間(0:00~0:00)",
                "points": ["このセクションの要点1", "このセクションの要点2", ...]
            }},
            {{
                "section_title": "セクション2のタイトル+時間(0:00~0:00)",
                "points": ["このセクションの要点1", "このセクションの要点2", ...]
            }}
        ]

        ```json
        {{
        // ここにJSONオブジェクトを生成
        }}
        ````
    """

    # ---------------------------------------------------------

    try:
        response = ollama.chat(
            model=current_model(),
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7, 'num_predict': 2000}
        )
        response_text = response['message']['content']
        
        json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]

        data = json.loads(json_str)

        # 従来通りのMarkdownファイル保存も行う
        md_output = f"# {data.get('title', 'No Title')}\n\n"
        md_output += f"## 概要\n{data.get('summary', '')}\n\n"
        md_output += f"## ハッシュタグ\n{' '.join(['#' + tag for tag in data.get('hashtags', [])])}\n\n"
        md_output += f"## キーワード\n{', '.join(data.get('keywords', []))}\n\n"
        md_output += f"## サムネイル文言\n> {data.get('thumbnail_text', '')}\n\n"
        md_output += "## 動画構成案\n"
        for i, section in enumerate(data.get('outline', [])):
            md_output += f"### {i+1}. {section.get('section_title', '')}\n"
            for point in section.get('points', []):
                md_output += f"- {point}\n"
            md_output += "\n"

        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = re.sub(r'[^\w\s-]', '', base_kw)[:20]
        safe_sug = re.sub(r'[^\w\s-]', '', suggestion)[:20]
        filename = f"{safe_base}_{safe_sug}_{ts}.md"
        path = os.path.join(save_dir, filename)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_output)
        st.success(f"アウトライン保存: {path}")

        return data

    except json.JSONDecodeError:
        st.error(f"JSONパースエラー。LLMの出力形式が不正な可能性があります。")
        st.text_area("LLMからの生レスポンス", response_text)
        return None
    except Exception as e:
        st.error(f"構成案生成エラー: {e}")
        return None

# --- メインUI ---
st.markdown("---")

# ステップ1: 検索ワード入力 & データ取得
base_query = st.text_input("検索ワードを入力してください：")
if st.button("YouTube からデータ取得"):
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        st.error("YouTube API キーを設定してください")
    else:
        with st.spinner("YouTube データを取得中..."):
            pattern = os.path.join(CACHE_DIR, f"{base_query}_*.csv") if CACHE_DIR else f"{base_query}_*.csv"
            cached = glob.glob(pattern)

            if cached:
                latest = max(cached, key=os.path.getmtime)
                df = pd.read_csv(latest)
                st.info(f"キャッシュを使用: {os.path.basename(latest)}")
            else:
                df = fetch_youtube_data(base_query)
                if not df.empty:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"{base_query}_{ts}.csv"
                    save_path = os.path.join(CACHE_DIR, fname) if CACHE_DIR else fname
                    df.to_csv(save_path, index=False)
                    st.success(f"保存完了: {fname}")
                    # 

            if df.empty:
                st.error("データが取得できませんでした。")
            else:
                st.dataframe(df.head(10))  # 最初の10件を表示
                st.info(f"全{len(df)}件のデータを取得しました")
                st.session_state['youtube_df'] = df

# ステップ2: 提案キーワード生成
if 'youtube_df' in st.session_state:
    st.markdown("---")
    st.subheader("ステップ2: 提案キーワード生成")
    
    num_keywords = st.slider("生成するキーワード数", 5, 20, 10)
    if st.button("提案キーワードを生成"):
        df_in = st.session_state['youtube_df']
        with st.spinner("キーワードを生成中..."):
            sug_df = generate_suggestions(df_in, query=base_query, num=num_keywords)
            if not sug_df.empty:
                st.session_state['suggestions'] = sug_df
                st.dataframe(sug_df)

# ステップ3: 構成生成
if 'suggestions' in st.session_state:
    st.markdown("---")
    st.subheader("ステップ3: 動画構成案を生成")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        purpose = st.text_input("動画の目的", "")
    with col2:
        target = st.text_input("ターゲット視聴者", "")
    
    selected_keywords = st.multiselect(
        "キーワードを選択:",
        st.session_state['suggestions']['keyword'].tolist()
    )
    with col3: 
        sections = st.number_input("セクション数", 1, 10, 4)
    
    with col4:
        duration = st.number_input("動画の長さ", 5, 30, step=5)
    
    if selected_keywords and st.button("構成案を生成"):
        for kw in selected_keywords:
            with st.spinner(f"「{kw}」の構成案を生成中..."):
                # 戻り値が辞書になる
                structured_outline = compose_structure(
                    base_query, kw,
                    sections=sections,
                    duration=duration,
                    target=target or "特になし",
                    purpose=purpose or "特になし"
                )
                
                # --- ここからが新しい表示部分 ---
                if structured_outline:
                    # expanderのタイトルにLLMが生成したタイトルを使用
                    expander_title = structured_outline.get('title', f'{base_query} + {kw}')
                    with st.expander(f"📝 **{expander_title}**", expanded=True):
                        
                        # st.tabsで見やすく情報を分類
                        tab1, tab2, tab3 = st.tabs(["**概要**", "**動画構成案**", "**メタ情報**"])

                        with tab1:
                            st.subheader("🖼️ サムネイル案")
                            # st.infoでサムネイル文言を目立たせる
                            st.info(f"**{structured_outline.get('thumbnail_text', 'N/A')}**")
                            
                            st.subheader("📄 概要文")
                            st.write(structured_outline.get('summary', ''))

                        with tab2:
                            st.subheader("🎬 動画構成案")
                            outline_sections = structured_outline.get('outline', [])
                            
                            # セクション数に応じてカラム数を動的に変更（最大3カラム）
                            num_cols = min(len(outline_sections), 1) 
                            if num_cols > 0:
                                cols = st.columns(num_cols)
                                for i, section in enumerate(outline_sections):
                                    with cols[i % num_cols]:
                                        # st.container(border=True)で各セクションをカード風に
                                        with st.container(border=True):
                                            st.markdown(f"**{i+1}. {section.get('section_title', 'No Title')}**")
                                            for point in section.get('points', []):
                                                st.markdown(f"- {point}")
                        
                        with tab3:
                            st.subheader("💡 キーワードとハッシュタグ")

                            # st.columnsで情報を並列に表示
                            col1, col2 = st.columns(2)
                            with col1:
                                 # st.metricで動画の長さを強調
                                st.metric(label="動画の長さ", value=f"{duration} 分")

                            # ハッシュタグをバッジ風に表示
                            hashtags = structured_outline.get('hashtags', [])
                            if hashtags:
                                st.markdown("**ハッシュタグ:** " + " ".join([f"`#{tag}`" for tag in hashtags]))
                            
                            # キーワードも同様に表示
                            keywords = structured_outline.get('keywords', [])
                            if keywords:
                                 st.markdown("**キーワード:** " + " ".join([f"`{kw}`" for kw in keywords]))

# 補足情報
st.sidebar.markdown("---")
st.sidebar.info("使用モデル: " + current_model())
st.sidebar.info("保存先: outlines/")
st.sidebar.caption("※ Ollamaが起動していることを確認してください")