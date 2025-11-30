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
st.set_page_config(page_title="YouTube + Gemma3 コラム案ジェネレータ", layout="centered")
st.title("YouTube + Gemma3 コラム案ジェネレータ")

# --- モデル切替UI（4b / 12b） ---
MODEL_OPTIONS = ("gemma3:latest", "gemma3:12b")

# 既定は12b（必要なら4bに変更OK）
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "gemma3:latest"

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

def parse_google_trends_csv(uploaded_file):
    """アップロードされたGoogle TrendsのCSVをパースする"""
    if uploaded_file is None:
        return None
    try:
        # ファイルのポインタを最初に戻してから読み込む
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.split('\n')

        top_keywords = []
        rising_keywords = []
        current_section = None

        top_header = 'TOP'
        rising_header = 'RISING'
        
        data_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if top_header in line:
                current_section = 'top'
                data_started = True
                continue
            elif rising_header in line:
                current_section = 'rising'
                data_started = True
                continue
            
            if not data_started or "カテゴリ:" in line:
                continue

            parts = line.split(',')
            if len(parts) >= 2:
                keyword = parts[0].strip('"')
                value = ','.join(parts[1:]).strip('"')
                if current_section == 'top':
                    top_keywords.append({'keyword': keyword, 'score': value})
                elif current_section == 'rising':
                    rising_keywords.append({'keyword': keyword, 'increase': value})

        df_top = pd.DataFrame(top_keywords)
        df_rising = pd.DataFrame(rising_keywords)
        
        if not df_top.empty:
            df_top.rename(columns={'score': 'importance'}, inplace=True)
        if not df_rising.empty:
            df_rising.rename(columns={'increase': 'importance'}, inplace=True)
        
        if not df_top.empty and not df_rising.empty:
            return pd.concat([df_top, df_rising]).dropna().reset_index(drop=True)
        elif not df_top.empty:
            return df_top
        elif not df_rising.empty:
            return df_rising
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"GoogleトレンドCSVの読み込みエラー: {e}")
        return None

def generate_suggestions(df, query, trend_df=None, num=5, save_path=None):
    if save_path:
        save_file = save_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"suggestions_{ts}.csv"

    # データの要約を作成（最初の10件のタイトルのみ使用）
    sample_titles = df.head(10)['title'].tolist()
    titles_text = "\n".join(sample_titles)
    
    # Googleトレンドデータの情報をプロンプトに追加
    trend_text = ""
    if trend_df is not None and not trend_df.empty:
        trend_text += "\n\nさらに、関連キーワードとしてGoogleトレンドで以下のデータが得られています。\n"
        trend_text += "これらは現在注目度が高い、または急上昇しているキーワードです。\n\n"
        trend_text += trend_df.to_string(index=False)
    
    prompt = f"""
あなたは優秀なコピーライターです。
検索キーワード「{query}」で見つかった動画タイトルの例：

{titles_text}
{trend_text}

これらの動画の共通テーマや、Googleトレンドで注目されているキーワードを考慮して、新しいトピックを{num}個生成してください。
特にGoogleトレンドのキーワードは重要度が高いので、積極的に含めてください。
地名・製品名・人物名などの固有名は避け、一般的なトピックにしてください。

以下のCSV形式で出力してください：
```csv
keyword
トピック1
トピック2
...
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

    conditions = [f"- {sections}つのセクション（見出しと本文）に分ける"]
    estimated_chars = duration * 400 
    conditions.append(f"- 全体で約{estimated_chars}字程度のボリューム")
    if target and target != "特になし":
        conditions.append(f"- ターゲット読者: {target}")
    if purpose and purpose != "特になし":
        conditions.append(f"- 記事の目的: {purpose}")
    conditions += ["- 専門用語は避け、初心者にも分かりやすい文章"]
    
    prompt = f"""
あなたは経験豊富なコンテンツライターです。
与えられたテーマと条件に基づき、読者の知的好奇心を満たすような、質の高いコラム記事を生成してください。

## テーマ: 「{full_topic}」

## 条件:
{chr(10).join(conditions)}

## 参考にするコラムの形式例：
これはあなたが目指すべき文章のスタイルです。各セクションは、単なる情報の羅列ではなく、背景やストーリーを感じさせる解説文にしてください。

### タイトル: テーマの核心をつき、読者がクリックしたくなるようなもの。
#### カテゴリー: 記事の内容を的確に表すキーワードを3つ。
##### 見出しと本文: 各セクションには、内容を要約した「見出し」と、背景やストーリーを感じさせる約300字程度の「本文」を作成してください。本文は単なる情報の羅列ではなく、読者に語りかけるようなスタイルで記述します。

## 出力形式:
以下のキーを持つJSON形式で出力してください。"body_text"は、参考例の（参考本文）のように、そのセクションで語るべき内容を読者に語りかけるような、200〜300字程度の本文として記述してください。

```json
{{
    "title": "記事のタイトル案",
    "category": "記事のカテゴリー（簡潔なものを3個程度）",
    "sections": [
        {{
            "heading": "セクション1の見出し",
            "body_text": "（ここにセクション1の本文を200〜300字で記述）"
        }},
        {{
            "heading": "セクション2の見出し",
            "body_text": "（ここにセクション2の本文を200〜300字で記述）"
        }}
    ]
}}
"""
    try:
        response = ollama.chat(
        model=current_model(),
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.7, 'num_predict': 4000} #文字数が増えるため少し増やす
        )
        response_text = response['message']['content']

        json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]

        data = json.loads(json_str)

        # 新しい形式でMarkdownファイルを保存
        md_output = f"# {data.get('title', 'No Title')}\n\n"
        md_output += f"カテゴリー: {data.get('category', 'N/A')}\n\n"
        
        for section in data.get('sections', []):
            md_output += f"## {section.get('heading', 'No Heading')}\n\n"
            md_output += f"{section.get('body_text', '')}\n\n" 
        
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = re.sub(r'[^\w\s-]', '', base_kw)[:20]
        safe_sug = re.sub(r'[^\w\s-]', '', suggestion)[:20]
        filename = f"{safe_base}_{safe_sug}_{ts}.md"
        path = os.path.join(save_dir, filename)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_output)

        # JSONファイルも新しい形式で保存
        json_filename = f"{safe_base}_{safe_sug}_{ts}.json"
        json_path = os.path.join(save_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        st.success(f"コラム保存完了: {filename}")

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
st.subheader("ステップ1: 分析の元となる情報を入力")
base_query = st.text_input("① 検索の軸となるキーワードを入力してください：", help="YouTubeで検索する際のキーワードです。")

# ▼▼▼ 変更点1: ラベルから「(任意)」を削除 ▼▼▼
uploaded_trend_file = st.file_uploader(
    "② Googleトレンドの関連キーワードCSVをアップロードしてください", 
    type=['csv'],
    help="Googleトレンドからダウンロードした「関連キーワード」のCSVファイルを指定します。"
)

# ファイルがアップロードされたらすぐにパースしてsession_stateに保存
if uploaded_trend_file:
    trend_df = parse_google_trends_csv(uploaded_trend_file)
    if trend_df is not None:
        # 成功メッセージはボタンを押した後に表示した方がUIがすっきりするため、ここではコメントアウトしても良い
        # st.success("GoogleトレンドCSVを読み込みました。")
        st.session_state['trend_df'] = trend_df
else:
    # ファイルが選択解除された場合に備えて、session_stateからキーを削除
    if 'trend_df' in st.session_state:
        del st.session_state['trend_df']


if st.button("YouTube からデータ取得"):
    # ▼▼▼ 変更点2: 必須チェック処理を追加 ▼▼▼
    # まずAPIキーをチェック
    if not API_KEY or "YOUR_API_KEY" in API_KEY:
        st.error("YouTube API キーを設定してください")
    # 次にキーワードとファイルの存在をチェック
    elif not base_query or not uploaded_trend_file:
        st.warning("キーワードの入力とCSVファイルのアップロードの両方が必要です。")
    # すべてOKの場合のみデータ取得処理に進む
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

            if df.empty:
                st.error("データが取得できませんでした。")
            else:
                st.dataframe(df.head(10))
                st.info(f"全{len(df)}件のデータを取得しました")
                st.session_state['youtube_df'] = df


# ステップ2: 提案キーワード生成
if 'youtube_df' in st.session_state:
    st.markdown("---")
    st.subheader("ステップ2: 提案キーワード生成")
    
    num_keywords = st.slider("生成するキーワード数", 5, 20, 10)
    if st.button("提案キーワードを生成"):
        df_in = st.session_state['youtube_df']
        trend_df_in = st.session_state.get('trend_df', None)
        with st.spinner("キーワードを生成中..."):
            sug_df = generate_suggestions(df_in, query=base_query, trend_df=trend_df_in, num=num_keywords)
            if not sug_df.empty:
                st.session_state['suggestions'] = sug_df
                st.dataframe(sug_df)

# ステップ3: 構成生成
if 'suggestions' in st.session_state:
    st.markdown("---")
    st.subheader("ステップ3: コラム構成案を生成")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        purpose = st.text_input("記事の目的", "")
    with col2:
        target = st.text_input("ターゲット読者", "")
    with col3: 
        sections = st.number_input("セクション数", 1, 10, 4)
    with col4:
        duration = st.number_input("記事のボリューム(分)", 3, 20, step=1, help="想定する読了時間を分で指定")
    
    selected_keywords = st.multiselect(
        "コラムを生成するキーワードを選択:",
        st.session_state['suggestions']['keyword'].tolist()
    )
    
    if selected_keywords and st.button("コラム構成案を生成"):
        for kw in selected_keywords:
            with st.spinner(f"「{kw}」のコラムを生成中..."):
                article_data = compose_structure(
                    base_query, kw,
                    sections=sections,
                    duration=duration,
                    target=target or "特になし",
                    purpose=purpose or "特になし"
                )
                
                if article_data:
                    # expander内に記事全体を表示
                    expander_title = article_data.get('title', f'{base_query} + {kw}')
                    with st.expander(f"📝 **生成されたコラム： {expander_title}**", expanded=True):
                        
                        st.markdown(f"## {article_data.get('title', 'No Title')}")
                        st.markdown(f"**カテゴリー:** {article_data.get('category', 'N/A')}")
                        st.markdown("---")
                        
                        for section in article_data.get('sections', []):
                            st.markdown(f"### {section.get('heading', 'No Heading')}")
                            st.write(section.get('body_text', ''))

# 補足情報
st.sidebar.markdown("---")
st.sidebar.info("使用モデル: " + current_model())
st.sidebar.info("保存先: outlines/")
st.sidebar.caption("※ Ollamaが起動していることを確認してください")