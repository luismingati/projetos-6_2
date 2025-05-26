import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Constantes
FEATURES = ['cores_vivas', 'versatilidade', 'conforto', 'formalidade', 'estampas']
CSV_PATH = 'sample_data/user_style_dataset_v2.csv' # Usar o novo dataset
OUTPUT_DIR = "models"
PIPELINE_FILE = os.path.join(OUTPUT_DIR, 'modelo_estilo.joblib')
CLUSTER_NAMES_FILE = os.path.join(OUTPUT_DIR, 'cluster_names.joblib')
RF_MODEL_FILE = os.path.join(OUTPUT_DIR, 'random_forest_model.joblib')

# Mapeamento base para nomear clusters (pode ser ajustado conforme a análise dos centroides)
BASE_NAME_MAP = {
    'cores_vivas': 'Colorista Vibrante',
    'versatilidade': 'Versátil Minimalista',
    'conforto': 'Casual Confortável',
    'formalidade': 'Profissional Moderno',
    'estampas': 'Aventureiro Fashion',
}

def train_and_evaluate_models(csv_path, features_list, output_dir):
    """
    Treina um pipeline de clustering (KMeans) e um classificador (RandomForest),
    e salva os modelos e nomes de clusters.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Carregando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df[features_list]

    # 1. Treinamento do Pipeline de Clustering (Scaler + KMeans)
    print("Treinando pipeline de clustering (StandardScaler + KMeans)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=5, n_init='auto', random_state=42))
    ])
    pipeline.fit(X)

    df['cluster_pred_kmeans'] = pipeline.predict(X) # Rótulos do KMeans
    
    silhouette_avg = silhouette_score(pipeline['scaler'].transform(X), df['cluster_pred_kmeans'])
    print(f"Silhouette Score para K-Means (k=5): {silhouette_avg:.3f}")

    # 2. Determinação dinâmica dos nomes dos clusters
    print("Determinando nomes dos clusters...")
    centroids_scaled = pipeline['kmeans'].cluster_centers_
    centroids_original_scale = pipeline['scaler'].inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original_scale, columns=features_list)

    cluster_names = {}
    # Para nomear, podemos encontrar a característica mais "distante" da média geral para cada cluster,
    # ou a de maior valor absoluto se as médias das features forem próximas de zero após o scaling.
    # Aqui, usamos a característica com maior valor no centroide original.
    # Uma abordagem mais robusta pode ser comparar com a média geral das features.
    overall_mean_features = X.mean() # Média das features no dataset original
    for idx, row in centroids_df.iterrows():
        # A característica dominante é aquela que mais se desvia positivamente da média geral,
        # ou simplesmente a de maior valor no centroide.
        # Para simplificar, pegamos a de maior valor no centroide.
        # dominant_feature = (row - overall_mean_features).idxmax() # Opção 1: desvio da média
        dominant_feature = row.idxmax() # Opção 2: maior valor no centroide
        cluster_names[idx] = BASE_NAME_MAP.get(dominant_feature, f"Cluster_{idx}")
    
    print(f"Nomes dos clusters determinados: {cluster_names}")
    df['cluster_nome'] = df['cluster_pred_kmeans'].map(cluster_names)

    # 3. Salvando o pipeline e os nomes dos clusters
    dump(pipeline, PIPELINE_FILE)
    dump(cluster_names, CLUSTER_NAMES_FILE)
    print(f"Pipeline salvo em: {PIPELINE_FILE}")
    print(f"Nomes dos clusters salvos em: {CLUSTER_NAMES_FILE}")

    # 4. Treinamento do RandomForestClassifier
    print("Treinando RandomForestClassifier...")
    y_rf = df['cluster_pred_kmeans'] # Usar os clusters do KMeans como target
    
    # Não é necessário train_test_split se o objetivo é apenas treinar o RF com todos os dados agrupados
    # Se fosse para avaliar o RF, um split seria necessário.
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_rf, test_size=0.25, stratify=y_rf, random_state=42
    # )

    # O RandomForest será treinado nas features originais (não escaladas),
    # pois a API receberá features não escaladas e o pipeline do RF pode incluir um scaler se necessário,
    # ou podemos escalar antes de passar para o RF.
    # Para consistência com o pipeline KMeans, vamos treinar RF nas features escaladas.
    # No entanto, a prática comum é treinar RF em features originais ou ter um pipeline para RF também.
    # O Colab treina RF em X_train (features originais). Vamos seguir isso.
    
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None, # Permitir que as árvores cresçam
        class_weight='balanced', # Lidar com desbalanceamento se houver
        random_state=42
    )
    # O pipeline já tem o scaler, então podemos usar pipeline.transform(X) para obter X_scaled
    # Mas o Colab usa X (features originais) para treinar o RF.
    # Se o RF for usado na API, e a API recebe features originais, então treinar em X faz sentido.
    # O scaler do pipeline KMeans será usado para transformar os dados antes de alimentar o RF na API.
    rf_model.fit(X, y_rf) # Treinando com features originais e rótulos do KMeans
    
    print(f"RandomForestClassifier treinado. Score no dataset de treino: {rf_model.score(X, y_rf):.3f}")
    
    # 5. Salvando o modelo RandomForest
    dump(rf_model, RF_MODEL_FILE)
    print(f"Modelo RandomForest salvo em: {RF_MODEL_FILE}")

    # 6. Visualizações (opcional, mas útil)
    generate_visualizations(pipeline['scaler'].transform(X), df['cluster_pred_kmeans'], df['cluster_nome'], output_dir)

    return pipeline, cluster_names, rf_model

def generate_visualizations(X_scaled_pca_input, labels_kmeans, cluster_names_map, output_dir):
    """
    Gera visualizações dos clusters usando PCA e histogramas.
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Aplicar PCA para visualização em 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_pca_input) # X_scaled_pca_input deve ser o X escalado

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.8) # Usar labels_kmeans
    plt.title('Clusters de Usuários (PCA)', fontsize=14)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    
    # Legenda com nomes dos clusters
    handles, _ = scatter.legend_elements()
    legend_labels = [cluster_names_map[i] for i in sorted(np.unique(labels_kmeans))]
    if len(handles) == len(legend_labels): # Garante que temos o mesmo número de handles e labels
         plt.legend(handles, legend_labels, title="Clusters")
    else: # Fallback se algo der errado com os handles/labels
        plt.colorbar(scatter, label='Cluster ID (KMeans)')

    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'pca_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Distribuição de usuários por cluster (usando nomes)
    plt.figure(figsize=(10, 7))
    # Mapear os labels numéricos para os nomes para o gráfico de barras
    named_labels = pd.Series(labels_kmeans).map(cluster_names_map)
    named_labels.value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribuição de Usuários por Perfil de Estilo', fontsize=14)
    plt.xlabel('Perfil de Estilo', fontsize=12)
    plt.ylabel('Quantidade de Usuários', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizações salvas em {viz_dir}")

if __name__ == "__main__":
    # Verifica se o CSV de dados existe
    if not os.path.exists(CSV_PATH):
        print(f"Arquivo de dados {CSV_PATH} não encontrado.")
        print("Por favor, execute generate_csv.py primeiro para criar o dataset.")
        import sys
        sys.exit(1)
        
    pipeline_model, c_names, random_forest_model = train_and_evaluate_models(CSV_PATH, FEATURES, OUTPUT_DIR)
    print("\nTreinamento e avaliação concluídos.")

    # Exemplo de como carregar e usar os modelos (para teste)
    from joblib import load
    loaded_pipeline = load(PIPELINE_FILE)
    loaded_cluster_names = load(CLUSTER_NAMES_FILE)
    loaded_rf_model = load(RF_MODEL_FILE)
    print("\nModelos carregados com sucesso para teste.")

    # Testar com um exemplo de features (escala 1-10)
    sample_features = np.array([[8, 6, 7, 3, 4]]) # Exemplo: Colorista, mas com outras notas
    
    # Para usar o RandomForest, ele foi treinado em features originais.
    # A API passará features originais para o RF.
    predicted_cluster_rf = loaded_rf_model.predict(sample_features)[0]
    predicted_profile_name_rf = loaded_cluster_names.get(predicted_cluster_rf)
    print(f"\nExemplo de predição com RandomForest:")
    print(f"Features: {sample_features[0]}")
    print(f"Cluster Predito (RF): {predicted_cluster_rf}")
    print(f"Nome do Perfil (RF): {predicted_profile_name_rf}")

    # Se quiséssemos usar o pipeline KMeans diretamente (não é o objetivo final, mas para comparação):
    # scaled_sample_features = loaded_pipeline.named_steps['scaler'].transform(sample_features)
    # predicted_cluster_kmeans = loaded_pipeline.named_steps['kmeans'].predict(scaled_sample_features)[0]
    # predicted_profile_name_kmeans = loaded_cluster_names.get(predicted_cluster_kmeans)
    # print(f"\nExemplo de predição com Pipeline K-Means (para comparação):")
    # print(f"Features: {sample_features[0]}")
    # print(f"Cluster Predito (KMeans): {predicted_cluster_kmeans}")
    # print(f"Nome do Perfil (KMeans): {predicted_profile_name_kmeans}")