import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings('ignore')

# ===== 1. 數據加載與基礎 ID 處理 =====
train = pd.read_csv("./Train.csv")
ss = pd.read_csv("./SampleSubmission.csv")

LABEL_MAP = {'free flowing': 0, 'light delay': 1, 'moderate delay': 2, 'heavy delay': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def extract_seg_id(s):
    try: return int(str(s).split('_')[2])
    except: return -1

def get_view_from_id(s):
    for i in range(1, 5):
        if f'#{i}' in str(s): return f'Norman Niles #{i}'
    return 'Unknown'

train['rating_num'] = train['congestion_enter_rating'].map(LABEL_MAP)
ss['time_segment_id'] = ss['ID'].apply(extract_seg_id)
ss['view_label'] = ss['ID'].apply(get_view_from_id)

# 標註並排序
train['is_train'] = 1
ss['is_train'] = 0
full_df = pd.concat([train, ss], ignore_index=True)
full_df = full_df.sort_values(['view_label', 'time_segment_id'])

# ===== 2. 核心：根據你提供的圖表建立「小時先驗特徵」 =====
# 根據圖片：10點後擁堵增加，11點 heavy delay 達到高峰
full_df['hour_idx'] = (full_df['time_segment_id'] % 48) # 假設 1 ID = 30min，48段為一天
# 估算大約的小時
full_df['approx_hour'] = (full_df['hour_idx'] / 2).astype(int) 

# 建立滯後特徵 (Lag 1-15)
for i in range(1, 16):
    full_df[f'lag_{i}'] = full_df.groupby('view_label')['rating_num'].shift(i)

full_df = full_df.fillna(0)

# ===== 3. 特徵提取 =====
def extract_final_features(df):
    X = pd.DataFrame()
    view_map = {'Norman Niles #1': 1, 'Norman Niles #2': 2, 'Norman Niles #3': 3, 'Norman Niles #4': 4}
    X['view'] = df['view_label'].map(view_map).fillna(0)
    X['hour'] = df['approx_hour']
    
    # 滯後特徵
    for i in range(1, 16):
        X[f'lag_{i}'] = df[f'lag_{i}']
    
    # 小時與擁堵的交互 (對應你給的分布圖)
    X['is_congestion_peak_hour'] = df['approx_hour'].apply(lambda x: 1 if 10 <= x <= 17 else 0)
    X['hour_view'] = X['hour'] * 10 + X['view']
    
    return X

train_final = full_df[full_df['is_train'] == 1]
test_final = full_df[full_df['is_train'] == 0]

X_train = extract_final_features(train_final)
y_train = train_final['rating_num']
X_test = extract_final_features(test_final)

# ===== 4. 解決不平衡：自定義樣本權重 =====
# 讓 10:00 - 17:00 之間的擁堵樣本權重更高
weights = compute_sample_weight(class_weight='balanced', y=y_train)
peak_mask = (X_train['is_congestion_peak_hour'] == 1).values
weights[peak_mask] *= 1.5 # 強化分布圖中顯示的高峰時段學習

# ===== 5. 訓練與動態機率調整 =====
model = XGBClassifier(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.02,
    objective='multi:softprob',
    num_class=4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, sample_weight=weights)

# 預測機率
probs = model.predict_proba(X_test)

# --- 後處理：根據分布圖調整 ---
# 如果時間在 10-17 點，且模型猶豫不決，微調偏向非 free flowing 類別
for i, hr in enumerate(X_test['hour']):
    if 10 <= hr <= 17:
        probs[i, 1:] *= 1.2 # 提升延遲類別的機率權重
        probs[i, 0] *= 0.8  # 壓低 free flowing 的機率

final_preds = np.argmax(probs, axis=1)
test_final['Target'] = [INV_LABEL_MAP[p] for p in final_preds]

# ===== 6. 輸出 =====
ss_output = ss[['ID']].merge(test_final[['ID', 'Target']], on='ID', how='left')
ss_output['Target_Accuracy'] = ss_output['Target']
ss_output.to_csv("./submission_hourly_prior.csv", index=False)

print("=" * 60)
print("📊 根據您提供的每小時分布圖修正完成！")
print(ss_output['Target'].value_counts())
print("=" * 60)