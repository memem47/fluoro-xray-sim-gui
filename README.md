# fluoro-xray-sim-gui

## 0. 最初のマイルストーン

- GitHubに新規リポジトリを作る
- Windowsでcloneして、VSCodeで開く
- Pythonアプリを起動すると「画像表示エリア」に X線が一様照射された風のノイズ画像（16-bitっぽいグレースケールでもOK）を表示
- 最初のコミットを作ってpush

- GUI
  - 最短で作りやすいPySide6(Qt)を使う（画像表示エリアが作りやすい）
  - 将来、動画生成・シミュレーション処理を足していきやすい構成にする

```powershell
cd C:\Users\<you>\source\repos
git clone https://github.com/<yourname>/fluoro-xray-sim-gui.git
cd fluoro-xray-sim-gui
code .
```



```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```