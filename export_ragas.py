import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables (API Key LangSmith)
load_dotenv()

client = Client()
PROJECT_NAME = "synapsa-pla"

print(f"Mengambil data dari project: {PROJECT_NAME}...")

# Filter hanya run dengan nama "ragas evaluation"
runs = list(
    client.list_runs(project_name=PROJECT_NAME, filter='eq(name, "ragas evaluation")')
)

data_penulisan = []

# Iterasi semua runs (biasanya RAGAS evaluation adalah run tersendiri)
for run in runs:
    # Filter hanya untuk RAGAS evaluation
    if run.name == "ragas evaluation" or "ragas" in run.name.lower():
        # Karena di RAGAS, output tersimpan sebagai dict (contoh: {"answer_relevancy": 0.8})
        output = run.outputs or {}
        if isinstance(output, list) and len(output) > 0:
            output = output[0]

        row = {
            "Run ID": str(run.id),
            "Waktu": run.start_time.strftime("%Y-%m-%d %H:%M:%S")
            if run.start_time
            else "",
            "Tipe Evaluasi": run.name,
            "Faithfulness": "",
            "Answer Relevancy": "",
            "Context Precision": "",
            "Context Recall": "",
        }

        if isinstance(output, dict):
            scores = output.get("scores", output)
            if isinstance(scores, list) and len(scores) > 0:
                scores = scores[0]

            if isinstance(scores, dict):
                row["Faithfulness"] = scores.get("faithfulness", "")
                row["Answer Relevancy"] = scores.get("answer_relevancy", "")
                row["Context Precision"] = scores.get("context_precision", "")
                row["Context Recall"] = scores.get("context_recall", "")

        # Coba ambil feedback kalau ada (meskipun RAGAS di sistem ini outputnya ke run.outputs)
        feedbacks = list(client.list_feedback(run_ids=[run.id]))
        for f in feedbacks:
            row[f.key] = f.score

        # Jika ada minimal satu metrik, masukkan ke data
        if row["Faithfulness"] != "" or row["Answer Relevancy"] != "":
            data_penulisan.append(row)

# Simpan ke CSV
if data_penulisan:
    df = pd.DataFrame(data_penulisan)
    # Reorder columns slightly for neatness

    # Save to CSV
    output_filename = "hasil_evaluasi_ragas.csv"
    df.to_csv(output_filename, index=False)
    print(
        f"Selesai! {len(data_penulisan)} baris data berhasil disimpan ke {output_filename}"
    )
else:
    print("Tidak ada data evaluasi RAGAS yang ditemukan.")
