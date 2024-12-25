from pathlib import Path

import pandas as pd

# 入力ディレクトリと出力ディレクトリを定義
base_path = Path(__file__).parent
input_path = base_path / "JPS-daprinfo/japanese personal information detect dataset"
output_path = (
    base_path / "JPS-daprinfo/japanese_personal_information_detect_dataset_csv"
)
output_path.mkdir(exist_ok=True, parents=True)

# 入力ディレクトリ内の.xlsxファイルを処理
for file_path in input_path.glob("*.xlsx"):
    try:
        # 出力ファイル名を設定
        output_file = output_path / file_path.with_suffix(".csv").name

        # ExcelをCSVに変換
        df = pd.read_excel(file_path, engine="openpyxl")
        df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"Converted {file_path.name} to {output_file.name}")
    except:
        print(f"Failed to convert {file_path.name}")
