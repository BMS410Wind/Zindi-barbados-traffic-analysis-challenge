import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ===== 設定區 (請務必修改這裡) =====
BASE_DIR = Path(__file__).resolve().parent

# 1. Anchor: 目前分數最高的檔案
ANCHOR_FILE = "submission_RF+GBM.csv"  # Sara提供

# 2. Challenger: 希望改進的新檔案
CHALLENGER_FILE = "0595.csv"  # Windspeak0109提交 ID:ZgEnwbDu

# 3. Supporters: 過去跑過還不錯的檔案 (用來驗證 Challenger 是否錯誤)
SUPPORT_FILES = [
    "submission_KNN+RF.csv",  # v2執行結果

    # "submission_old_0.53.csv" # 分數太低的建議不要放，會變成雜訊
]

# 標籤映射
LABEL_MAP = {'free flowing': 0, 'light delay': 1,
             'moderate delay': 2, 'heavy delay': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

print("啟動差異驗證修正器 (Consensus Corrector)...")
print("=" * 60)

# ===== 1. 讀取資料 =====


def load_pred(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {filename}")
    df = pd.read_csv(path)
    # 確保依照 ID 排序
    df = df.sort_values('ID').reset_index(drop=True)
    return df['Target'].map(LABEL_MAP).fillna(0).astype(int).values, df


print(f"正在讀取 Anchor (基準): {ANCHOR_FILE}")
anchor_preds, df_template = load_pred(ANCHOR_FILE)

print(f"正在讀取 Challenger (新模型): {CHALLENGER_FILE}")
challenger_preds, _ = load_pred(CHALLENGER_FILE)

supporters_preds = []
for f in SUPPORT_FILES:
    print(f"正在讀取 Supporter (輔助): {f}")
    p, _ = load_pred(f)
    supporters_preds.append(p)

supporters_matrix = np.array(supporters_preds)

# ===== 2. 執行修正邏輯 =====
final_preds = anchor_preds.copy()
change_count = 0
improved_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 記錄改成了什麼

total_rows = len(anchor_preds)

for i in range(total_rows):
    base = anchor_preds[i]
    new = challenger_preds[i]

    # 如果新模型跟基準一樣，那就不用浪費時間判斷
    if base == new:
        continue

    # --- 差異發生！開始仲裁 ---
    # 取得所有 Supporter 在這個 ID 的看法
    support_votes = supporters_matrix[:, i]

    # 策略 A: 強力驗證 (Strong Validation)
    # 只有當 "新模型" 的看法，獲得 "至少一半輔助模型" 的支持時，才推翻基準
    # 例如: Base=0, New=3. Supporters=[3, 3, 2].
    # New(3) 獲得了兩個 3 的支持 -> 修改生效

    match_count = np.sum(support_votes == new)
    threshold = len(SUPPORT_FILES) / 2  # 過半數門檻

    if match_count >= threshold:
        # 批准修改！
        final_preds[i] = new
        change_count += 1
        improved_counts[new] += 1

    # 策略 B (可選): 針對 Heavy Delay 的特別召回
    # 如果 Base=Moderate(2), New=Heavy(3)，且只要有任一個 Supporter 也是 3
    # 我們就冒險改上去 (因為 Heavy 分數高)
    elif base == 2 and new == 3 and np.any(support_votes == 3):
        final_preds[i] = 3
        change_count += 1
        improved_counts[3] += 1

# ===== 3. 輸出結果 =====
df_template['Target'] = [INV_LABEL_MAP[p] for p in final_preds]
df_template['Target_Accuracy'] = df_template['Target']


timestamp = time.strftime("%Y%m%d_%H%M%S")
output_filename = f"submission_corrected_consensus_{timestamp}.csv"
output_path = BASE_DIR / output_filename
df_template.to_csv(output_path, index=False)

print("=" * 60)
print(f"修正完成！")
print(f"總共修改了 {change_count} 筆預測 (佔總數 {change_count/total_rows:.1%})")
print(f"修改方向分佈: {improved_counts}")
print(f"檔案已儲存至: {output_path}")
print("=" * 60)
