import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import soundfile as sf

class FSDDDataset(Dataset):
    def __init__(self, data_dir):
        #data_dirディレクトリ内にある.wav音声ファイルを取り出す。
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

        #1次元の波を時間×周波数の2次元データに変換する
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = 8000, #サンプリングレート。8㎑で保存する。
            n_fft = 1024, #窓サイズ。フーリエ変換を行うために音声を切り取る枠サイズ。
            hop_length = 512, #ずらし幅。窓を半分ずつ重ねながらスライドさせていく。
            n_mels = 64 #メル周波数ビン数。AIに入力する画像の高さ。メル尺度で圧縮
        )

        # 振幅をデシベル（対数スケール）に変換
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        #すべてのデータを32フレームにそろえる
        self.target_length = 32

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        #読み込むファイルのパスを作成
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        #インデックスを分離することで正解ラベルをつくる
        label = int(self.file_list[idx].split('_')[0])

        # 最新版torchaudioのバグを回避するため、直接soundfileで読み込む
        waveform_numpy, _ = sf.read(file_path, dtype='float32')
        # numpy配列をPyTorchのテンソルに変換し、[1(チャンネル数), 時間] の形に整える
        waveform = torch.from_numpy(waveform_numpy).unsqueeze(0)

        #短時間フーリエ変換
        mel_spec = self.mel_spectrogram(waveform)
        #変換した2次元データの振幅をデシベルに
        mel_spec_db = self.amplitude_to_db(mel_spec)

        #長さをそろえる処理
        current_length = mel_spec_db.shape[-1]
        
        if current_length < self.target_length:
            # 【短い場合】足りない分だけ、後ろ（右側）に 0 を埋める（パディング）
            pad_amount = self.target_length - current_length
            
            # F.pad(テンソル, (左に足す数, 右に足す数)) というルール
            mel_spec_db = F.pad(mel_spec_db, (0, pad_amount))
            
        elif current_length > self.target_length:
            # 【長い場合】はみ出た部分をバッサリ切り捨てる（クロップ）
            # [..., :self.target_length] は「最後の次元だけを指定の長さで切る」という書き方
            mel_spec_db = mel_spec_db[..., :self.target_length]
            
        return mel_spec_db, label
    
class AudioCNN(nn.Module):
    def __init__(self):
        #親クラスを継承
        super(AudioCNN, self).__init__()
        #最初の畳み込み層：画像からエッジやパターン特徴を抽出
        self.conv1 = nn.Conv2d(in_channels = 1,#入力がモノクロである。音声は1
                               out_channels = 16,#16個のフィルターで16個の特徴パターンを出力
                               kernel_size = 3,#3×3のフィルター
                               stride = 1,#1マスずつスライド
                               padding = 1)
        #プーリング層：計算量を減らすずれをなくす
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)
        #2つ目の畳み込み層：conv1で抽出された特徴を組合して、より複雑な特徴を捉える
        self.conv2 = nn.Conv2d(16,32,3,1,1)#出力チャンネルが32に増えている
        #全結合層：抽出された特徴を1列に並べ、0~9を判断する
        self.fc1 = nn.Linear(32*16*8,128)
        #最終出力層：0~9のそれぞれの確立を出力する
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#畳み込み
        x = self.pool(F.relu(self.conv2(x)))#畳み込み
        x = x.view(x.size(0), -1) # 1次元に平滑化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)#プーリング
        return x
    
#計算をCPUで行うかGPUで行うかを自動判定    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN().to(device)
#損失関数：AIのずれと正解のずれを計算する。
criterion = nn.CrossEntropyLoss()
#最適化手法：ネットワークをどう更新するか決めるアルゴリズム。
#Adamが人気のアルゴリズムで、lrは学習率。
optimizer = optim.Adam(model.parameters(),lr = 0.001)

#実際に学習ループをまわす
all_dataset = FSDDDataset(data_dir="./recordings")
# 訓練データ(80%)とテストデータ(20%)に分割
train_size = int(0.8 * len(all_dataset))
test_size = len(all_dataset) - train_size
train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#1.テストを解く
num_epochs = 10#問題集を十周
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for epoch in range(num_epochs):
    #小分けにして解く。inputsがデータ、labelsが正解の数字
    for inputs,labels in train_loader:
        #GPU or CPUに正解ラベルを移動
        inputs, labels = inputs.to(device),labels.to(device)

        optimizer.zero_grad()      # 勾配の初期化 / 反省点メモを白紙に
        outputs = model(inputs)    # 順伝播 / テストを解く
        loss = criterion(outputs, labels) # 誤差計算 / 答え合わせ
        loss.backward()            # 逆伝播 / どこ間違えたか分析
        optimizer.step()           # パラメータ更新 / 復習して賢く
    #損失関数が1エポックでどれだけ減ったかを表示
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}") 

# ------------------------------------------
# 学習が終わったあとの「本番テスト」
# ------------------------------------------
print("テストデータでAIの実力を測定中...")

# モデルを「テストモード」に切り替える（一部の挙動を評価用に固定するため）
model.eval() 

correct = 0 # 正解した数
total = 0   # 出題した全問題数

# テスト中は「学習（反省）」をしないので、メモリ節約のため勾配計算をストップする
with torch.no_grad(): 
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # AIに予測させる
        outputs = model(inputs)
        
        # 10個の数字の確率のうち、一番自信がある（数値が大きい）ものを選ぶ
        _, predicted = torch.max(outputs.data, 1) 
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 正解率の計算
accuracy = 100 * correct / total
print(f"🎉 最終テスト正解率: {accuracy:.2f} %")

# ------------------------------------------
# 学習した「脳の記憶」をセーブする
# ------------------------------------------
# model.state_dict() が「学習で最適化されたパラメータ（重み）」のデータです
save_path = "fsdd_cnn_model.pth"
torch.save(model.state_dict(), save_path)

print(f"🎉 AIの記憶を {save_path} に保存しました！")