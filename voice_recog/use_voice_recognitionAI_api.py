import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
# ==========================================
# 1. AIの「脳の形（設計図）」を教える
# （作った時と全く同じものを書きます）
# ==========================================
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32 * 16 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#セーブデータ（記憶）のロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ① まず「空っぽの脳」を作る
model = AudioCNN().to(device)
# ② そこにセーブデータ（記憶）を流し込む
# ※weights_only=True は最近のPyTorchで推奨される安全設定です
model.load_state_dict(torch.load("fsdd_cnn_model.pth", weights_only=True))
# ③ テスト（本番）モードに切り替える
model.eval()

# librosaを使って、どんな録音でも強制的に「8000Hz」で読み込む
waveform_numpy, _ = librosa.load("my_voice.wav", sr=8000)
waveform = torch.from_numpy(waveform_numpy).unsqueeze(0).to(device) # AI用のテンソルに変換

# AIが学習した時と「全く同じ設定」のスペクトログラム変換器を作る
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=8000, n_fft=1024, hop_length=512, n_mels=64
).to(device)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

# 音声を画像に変換
mel_spec = mel_spectrogram(waveform)
mel_spec_db = amplitude_to_db(mel_spec)

# AIの受付サイズ「長さ32」に無理やり合わせる
target_length = 32
current_length = mel_spec_db.shape[-1]
if current_length < target_length:
    mel_spec_db = F.pad(mel_spec_db, (0, target_length - current_length))
elif current_length > target_length:
    mel_spec_db = mel_spec_db[..., :target_length]

# バッチサイズ1の次元を追加 [1(枚), 1(モノクロ), 64(縦), 32(横)]
input_tensor = mel_spec_db.unsqueeze(0)

# ==========================================
# 3. AIに判定させる！
# ==========================================
with torch.no_grad():
    output = model(input_tensor)
    
    # 確率をパーセントに変換して見やすくする
    probabilities = F.softmax(output, dim=1)[0] * 100
    predicted_digit = torch.argmax(output, 1).item()

print("\nAIの判定結果 ")
print(f"あなたが言った数字は... 【 {predicted_digit} 】 です\n")

print("📊 各数字である確率:")
for i in range(10):
    print(f"数字 {i} : {probabilities[i]:.1f} %")




