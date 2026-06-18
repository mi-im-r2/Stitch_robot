import librosa
import soundfile as sf

# 1. 音声データの読み込み
# sr=None は「元のサンプリングレートを維持する」という指定です
print("音声を読み込んでいます...")
y, sr = librosa.load("test.wav", sr=None)

# 2. ピッチシフト（声の高さを変える）
# n_steps = 半音単位での変化量。+4で少し高く、-4で低くなります。
# 内部的にSTFT（短時間フーリエ変換）と位相操作が行われています。
print("変換処理中...")
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=5)

# 3. 変換した音声の書き出し
output_file = "shifted_voice.wav"
sf.write(output_file, y_shifted, sr)
print(f"完了しました！ {output_file} を再生してみてください。")