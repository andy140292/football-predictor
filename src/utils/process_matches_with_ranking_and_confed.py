from utils.add_ranking_fifa_to_matches import add_ranking_fifa_to_matches
from utils.confederation_mapping import add_confederation_to_matches
import pandas as pd
from datetime import datetime
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from supabase import create_client, Client
from dotenv import load_dotenv
import gzip
from datetime import datetime, timedelta

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. Descargar el dataset desde Kaggle
def download_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset = "martj42/international-football-results-from-1872-to-2017"
    dest_dir = "kaggle_data"
    os.makedirs(dest_dir, exist_ok=True)
    api.dataset_download_files(dataset, path=dest_dir, unzip=True)
    print("✅ Dataset descargado exitosamente.")

# 2. Cargar y procesar los datos
def process_matches():
    # 3. Leer partidos descargados
    matches_path = os.path.join("kaggle_data", "results.csv")
    matches_df = pd.read_csv(matches_path)

    # 4. Agregar confederación y ranking
    updated_df_with_confed = add_confederation_to_matches(matches_df)
    matches_df = updated_df_with_confed

    # 5. Leer ranking FIFA actualizado
    ranking_df = pd.read_csv("data/ranking_fifa_2025.csv")
    updated_df_with_ranking = add_ranking_fifa_to_matches(updated_df_with_confed, ranking_df)
    matches_df = updated_df_with_ranking

    # 6. Guardar archivo versionado
    month_str = (datetime.today() - timedelta(days=30)).strftime("%Y_%m")
    output_file = f"data/matches_{month_str}.csv"
    matches_df.to_csv(output_file, index=False)
    print(f"✅ Archivo procesado y guardado como: {output_file}")

def upload_to_supabase():
    today = datetime.today().strftime("%Y_%m")
    file_path = f"data/matches_{today}.csv"
    storage_path = f"matches_{today}.csv"
    compressed_path = f"data/matches_{today}.csv.gz"

    with open(file_path, "rb") as f_in:
        with gzip.open(compressed_path, "wb") as f_out:
            f_out.writelines(f_in)

    with open(compressed_path, "rb") as f:
        supabase.storage.from_("match-datasets").upload(
            f"matches_{today}.csv.gz",
            f,
            file_options={"content-type": "application/gzip", "cache-control": "3600"}
        )
    print(f"📤 Archivo {file_path} subido a Supabase como {storage_path}")

# Ejecutar el flujo completo
if __name__ == "__main__":
    download_dataset()
    process_matches()