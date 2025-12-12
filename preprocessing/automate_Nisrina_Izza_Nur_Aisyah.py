import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_data(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Data berhasil dimuat dari {path}. Ukuran: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di {path}")
        return None

def preprocessing_data(df):
    df_clean = df.copy()

    # 1. Target Engineering
    if 'Improvement_Score' in df_clean.columns:
        threshold = df_clean['Improvement_Score'].median()
        print(f"Threshold Median yang digunakan: {threshold}")

        # Bagi data berdasarkan nilai tengah
        df_clean['Effectiveness'] = df_clean['Improvement_Score'].apply(lambda x: 1 if x > threshold else 0)

    # 2. Feature Selection
    cols_to_drop = ['Patient_ID', 'Drug_Name', 'Improvement_Score']
    cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    # 3. Encoding - Gender (Label Encoding)
    if 'Gender' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Gender'] = le.fit_transform(df_clean['Gender'])

    # 4. Encoding - Condition & Side_Effects (One-Hot Encoding)
    for col in ['Condition', 'Side_Effects']:
        if col in df_clean.columns:
            top_categories = df[col].value_counts().nlargest(10).index
            df_clean[col] = df_clean[col].apply(lambda x: x if x in top_categories else 'Other')

    df_clean = pd.get_dummies(df_clean, columns=['Condition', 'Side_Effects'], prefix=['Cond', 'SE'])


    bool_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'bool']
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    # 5. Scaling (StandardScaler)
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Dosage_mg', 'Treatment_Duration_days']
    
    exist_num_cols = [c for c in numerical_cols if c in df_clean.columns]
    if exist_num_cols:
        df_clean[exist_num_cols] = scaler.fit_transform(df_clean[exist_num_cols])

    return df_clean

def save_data(df, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Data bersih berhasil disimpan di: {output_path}")


if __name__ == "__main__":
    print("🚀 Memulai Otomatisasi Preprocessing...")
    script_dir = os.path.dirname(os.path.abspath(__file__))


    project_root = os.path.dirname(script_dir)


    input_csv = os.path.join(project_root, "drug_dataset_raw", "real_drug_dataset.csv")

    output_csv = os.path.join(script_dir, "drug_dataset_preprocessing", "train_clean.csv")


    print(f"📂 Lokasi Script : {script_dir}")
    print(f"📂 Lokasi Root   : {project_root}")
    print(f"🔍 Mencari Input : {input_csv}")
    print(f"💾 Target Output : {output_csv}")
    print("-" * 30)

    if os.path.exists(input_csv):
        df_raw = get_data(input_csv)

        if df_raw is not None:
            df_processed = preprocessing_data(df_raw)
            save_data(df_processed, output_csv)
            print("✅ SUKSES! File train_clean.csv berhasil dibuat.")
    else:
        print("❌ ERROR: File input masih belum ditemukan!")
        print("   Pastikan nama file di folder 'drug_dataset_raw' benar-benar 'real_drug_dataset.csv'")
