# Voice Recognition AI (0-9 Digits)

自作のCNN（畳み込みニューラルネットワーク）を用いて、0から9までの音声数字を判別するAIモデルです。

## 概要
音声データセットを学習させ、自分の声でも数字を正しく認識できるかをテストするためのプロジェクトです。
機械学習の基礎的な実装から、実際の音声データの前処理、推論までを一貫して行っています。

## ファイル構成
* `voice_recognitionAI.py`: CNNモデルの構築と学習を行うメインスクリプト
* `use_voice_recognition_api.py`: 学習済みモデルを読み込み、自分の声(`my_voice.wav`)を入力して判別テストを行うスクリプト
* `fsdd_cnn_model.pth`: 学習済みのモデルファイル

## 実行方法

### 1. データセットの準備
このリポジトリには学習用の音声データが含まれていません。
実行する前に、[FSDD (Free Spoken Digit Dataset)など] からデータセットをダウンロードし、プロジェクトの直下に `recordings` という名前のフォルダを作成して配置してください。

### 2. 環境構築
必要なライブラリをインストールします。
```bash
pip install torch torchaudio # 他にlibrosaなどがあれば追記

### 3.学習の実行
新たにモデルを学習させる場合は以下のコマンドを実行します。

Bash
python voice_recognitionAI.py

### 4. 自分の声でテスト
テスト用の音声（my_voice.wav）をプロジェクト直下に配置し、以下を実行します。

Bash
python use_voice_recognition_api.py