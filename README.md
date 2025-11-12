# streamlit.app
生成AIを用いてyoutube動画の構成案を作成するツールです


## 🛠 セットアップ手順

### ① ZIPをダウンロード
1. GitHub右上の「Code ▾」→「Download ZIP」  
2. ZIPを右クリック → 「すべて展開」  
3. 展開したフォルダを開く


### ② Pythonをインストール
1. [Python公式サイト](https://www.python.org/downloads/) にアクセス。  
2. **Python 3.11.x** を選択してダウンロード。  
3. インストール時に必ず「**Add Python to PATH**」にチェックを入れてください。

インストール後、動作確認：
```bash
python --version
```

### ③ Ollamaをインストール

- [Ollama公式サイト](https://ollama.com/download) からインストール  
- 起動確認：

```bash
ollama serve
```


### ④ モデルをダウンロード

初回のみ実行：

```bash
ollama pull llama3:4b
ollama pull llama3:12b
```


### ⑤ ライブラリをインストール

1. 展開したフォルダを開く  
2. アドレスバーに `cmd` と入力して **Enter**  
3. 次を実行：

```bash
pip install -r requirements.txt
```


### ⑥ アプリを起動

```bash
streamlit run streamlit_gemma_0916_json.py
```

🌐 **ブラウザで自動的に開きます：**  
[http://localhost:8501](http://localhost:8501)



