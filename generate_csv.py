import os
import numpy as np
import pandas as pd

# Simular dataset de usuários fictícios com resposta de 1 a 10 e label de cluster
np.random.seed(42)

# Perfis e suas características médias (escala 1-10)
MEANS = {
    0: {'cores_vivas': 9, 'versatilidade': 5, 'conforto': 5, 'formalidade': 5, 'estampas': 5},  # Colorista Vibrante
    1: {'cores_vivas': 5, 'versatilidade': 9, 'conforto': 5, 'formalidade': 5, 'estampas': 5},  # Versátil Minimalista
    2: {'cores_vivas': 5, 'versatilidade': 5, 'conforto': 9, 'formalidade': 5, 'estampas': 5},  # Casual Confortável
    3: {'cores_vivas': 5, 'versatilidade': 5, 'conforto': 5, 'formalidade': 9, 'estampas': 5},  # Profissional Moderno
    4: {'cores_vivas': 5, 'versatilidade': 5, 'conforto': 5, 'formalidade': 5, 'estampas': 9},  # Aventureiro Fashion
}

rows = []
for cluster_real, m_values in MEANS.items():
    for _ in range(45):  # 45 amostras por cluster
        sample = {
            k: int(np.clip(np.random.normal(loc=v, scale=1.2), 1, 10))
            for k, v in m_values.items()
        }
        sample['cluster_real'] = cluster_real  # Adiciona o cluster real para referência
        rows.append(sample)

df = pd.DataFrame(rows)
df.insert(0, 'user_id', [f'U{i+1:03d}' for i in range(len(df))])

os.makedirs('sample_data', exist_ok=True)

# Salvar CSV para download
csv_path = 'sample_data/user_style_dataset_v2.csv'
df.to_csv(csv_path, index=False)

print(f"Dataset gerado e salvo em: {csv_path}")
print("Primeiras linhas do dataset:")
print(df.head())