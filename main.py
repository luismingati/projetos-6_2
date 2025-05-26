from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
import pandas as pd
import sqlite3
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
from openai import OpenAI
import logging
from joblib import load
import numpy as np
from dotenv import load_dotenv
import asyncio
from supabase import Client, create_client
from datetime import datetime
import uuid
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clothes API", description="API para recomendação de roupas e classificação de perfil de moda")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "clothes-images")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("As variáveis SUPABASE_URL e SUPABASE_ANON_KEY devem estar definidas no .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(BASE_MODEL_DIR, "modelo_estilo.joblib")
CLUSTER_NAMES_PATH = os.path.join(BASE_MODEL_DIR, "cluster_names.joblib")
RF_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "random_forest_model.joblib")

try:
    df_clothes_dataset = pd.read_parquet( 
        "hf://datasets/wbensvage/clothes_desc/data/my_clothes_desc.parquet"
    )
except Exception as e:
    logger.error(f"Erro ao carregar o dataset de roupas: {e}")
    df_clothes_dataset = None

pipeline_model = None
cluster_names_map = None
rf_model = None

def load_all_models():
    global pipeline_model, cluster_names_map, rf_model
    try:
        pipeline_model = load(PIPELINE_PATH)
        cluster_names_map = load(CLUSTER_NAMES_PATH)
        rf_model = load(RF_MODEL_PATH)
        
        logger.info("Todos os modelos carregados com sucesso.")
        logger.info(f"Pipeline: {type(pipeline_model)}")
        logger.info(f"Nomes dos Clusters: {cluster_names_map}")
        logger.info(f"Modelo RandomForest: {type(rf_model)}")

        return pipeline_model, cluster_names_map, rf_model
        
    except FileNotFoundError as e:
        logger.error(f"Arquivos de modelo não encontrados: {e}. Execute o treinamento primeiro.")
        return None, None, None
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        return None, None, None

DATABASE_URL = "clothes_ids.db"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("A variável de ambiente OPENAI_API_KEY não está definida.")
client = OpenAI(api_key=api_key)

def init_db():
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS saved_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL UNIQUE,
            text TEXT,
            cores_vivas INTEGER,  
            versatilidade INTEGER,
            conforto INTEGER,
            formalidade INTEGER,
            estampas INTEGER,
            cluster_id INTEGER, 
            cluster_name TEXT  
        )
        '''
    ) 
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS uploaded_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            supabase_path TEXT NOT NULL UNIQUE,
            public_url TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            content_type TEXT
        )
        '''
    )
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS garment_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            garment_dataset_id TEXT NOT NULL UNIQUE, -- Corresponde ao 'idx' do seu dataset ou ao número na imagem (ex: '0', '1', ..., '99')
            supabase_url TEXT NOT NULL,
            original_filename TEXT, -- Ex: 'img_0.jpg'
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    conn.commit()
    conn.close()

def ensure_model_dir():
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_db()
    ensure_model_dir()
    global pipeline_model, cluster_names_map, rf_model
    pipeline_model, cluster_names_map, rf_model = load_all_models() 
    if not all([pipeline_model, cluster_names_map, rf_model]):
        logger.warning("Um ou mais modelos não puderam ser carregados no startup. Alguns endpoints podem não funcionar.")
    else:
        logger.info("Aplicação inicializada com sucesso e modelos carregados.")


class ImageIDs(BaseModel):
    ids: List[str]

class ClothingFeatures(BaseModel):
    cores_vivas: int = Field(..., ge=1, le=10, description="Classificação de 1 a 10 para 'Gosto de cores vivas'")
    versatilidade: int = Field(..., ge=1, le=10, description="Classificação de 1 a 10 para 'Prefiro peças versáteis que combinam com tudo'")
    conforto: int = Field(..., ge=1, le=10, description="Classificação de 1 a 10 para 'Busco conforto acima de estilo'")
    formalidade: int = Field(..., ge=1, le=10, description="Classificação de 1 a 10 para 'Gosto de roupas mais formais'")
    estampas: int = Field(..., ge=1, le=10, description="Classificação de 1 a 10 para 'Me atraem estampas chamativas'")

class UserProfileInput(BaseModel):
    features: ClothingFeatures
    
class ProfileResponse(BaseModel):
    cluster_id: int 
    profile_name: str
    description: str

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[Dict[str, str]]
    failed_files: List[Dict[str, str]]


class OpenAIClothTitleDescriptionResponse(BaseModel):
    title: str = Field(..., description="Título da peça de roupa, ex: Calça Jogger")
    description: str = Field(..., description="Breve descrição do produto em português do Brasil")

class ClothItemDetail(BaseModel):
    id: str
    image_url: str             # Alterado de image_representation para image_url
    title: str
    description: str
    original_text: Optional[str] = None


def get_features_schema():
    schema = ClothingFeatures.model_json_schema()
    for prop_schema in schema.get("properties", {}).values():
        prop_schema.pop("minimum", None)
        prop_schema.pop("maximum", None) 

    schema["additionalProperties"] = False 
    return schema

PROFILE_DESCRIPTIONS = {
    'Colorista Vibrante': "Você adora se expressar com cores intensas e ousadas. Seu guarda-roupa é uma explosão de tons vibrantes, e você não tem medo de experimentar combinações cromáticas que chamam a atenção.",
    'Versátil Minimalista': "Sua preferência é por peças atemporais e versáteis que formam uma base sólida para diversos looks. Você valoriza a simplicidade elegante, linhas clean e a capacidade de adaptar suas roupas a diferentes ocasiões com poucos ajustes.",
    'Casual Confortável': "O conforto é sua prioridade máxima. Você busca peças aconchegantes, tecidos macios e modelagens que permitam liberdade de movimento, sem abrir mão de um estilo casual e despojado para o dia a dia.",
    'Profissional Moderno': "Você se veste para o sucesso, preferindo peças com um toque de formalidade, mas sempre alinhadas com as tendências contemporâneas. Seu estilo transmite confiança e sofisticação no ambiente de trabalho e além.",
    'Aventureiro Fashion': "Seu estilo é marcado por estampas criativas, texturas interessantes e uma vontade de explorar o novo. Você vê a moda como uma forma de aventura e autoexpressão, sempre em busca de peças únicas que contem uma história."
}

async def gen_title_description(text_value: str, image_id: str) -> dict:
    """Faz a chamada ao OpenAI e retorna um dict com title e description."""
    system_prompt = (
        "Você é um assistente de IA especialista em moda. Sua tarefa é analisar a descrição "
        "e gerar um JSON com 'title' e 'description' em PT-BR."
    )
    user_prompt = f"Descrição da peça: '{text_value}'"
    try:
        resp = await client.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = resp.choices[0].message.content
        parsed = OpenAIClothTitleDescriptionResponse.model_validate_json(data)
        return {"title": parsed.title, "description": parsed.description}
    except AttributeError:
        result = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = result.choices[0].message.content
        parsed = OpenAIClothTitleDescriptionResponse.model_validate_json(data)
        return {"title": parsed.title, "description": parsed.description}

# ... (importações e configurações existentes) ...

async def upload_local_garment_images_to_supabase(
    local_image_directory: str = "images", 
    image_prefix: str = "img_", 
    image_extension: str = ".jpg", 
    num_images: int = 1000 # Suas 100 imagens leves
):
    """
    Faz upload de imagens locais (ex: img_0.jpg, ..., img_99.jpg) para o Supabase
    e salva suas URLs públicas no banco de dados SQLite.
    """
    if not os.path.isdir(local_image_directory):
        logger.error(f"Diretório de imagens locais '{local_image_directory}' não encontrado.")
        return {"error": f"Diretório '{local_image_directory}' não encontrado."}

    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    uploaded_count = 0
    failed_uploads = []

    for i in range(num_images):
        original_filename = f"{image_prefix}{i}{image_extension}" # ex: img_0.jpg
        local_file_path = os.path.join(local_image_directory, original_filename)
        garment_dataset_id = str(i) # ID que será usado para buscar no dataset/tabela

        if not os.path.exists(local_file_path):
            logger.warning(f"Arquivo de imagem local não encontrado: {local_file_path}")
            failed_uploads.append({"filename": original_filename, "reason": "Arquivo local não encontrado"})
            continue

        try:
            with open(local_file_path, "rb") as f:
                file_content = f.read()
            
            # Determina o content-type (simples, pode ser melhorado)
            content_type = "image/jpeg" if image_extension.lower() == ".jpg" else "image/png"
            
            # Cria um nome de arquivo único para o Supabase para evitar colisões
            supabase_filename = f"garments/{generate_unique_filename(original_filename)}"

            # Upload para o Supabase
            upload_result = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(
                path=supabase_filename,
                file=file_content,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                },
            )
            
            public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(supabase_filename)
            
            try:
                cursor.execute(
                    """
                    INSERT INTO garment_images (garment_dataset_id, supabase_url, original_filename)
                    VALUES (?, ?, ?)
                    ON CONFLICT(garment_dataset_id) DO UPDATE SET
                    supabase_url = excluded.supabase_url,
                    original_filename = excluded.original_filename,
                    uploaded_at = CURRENT_TIMESTAMP
                    """,
                    (garment_dataset_id, public_url, original_filename)
                )
                conn.commit()
                uploaded_count += 1
                logger.info(f"Upload bem-sucedido e DB atualizado para: {original_filename} -> {public_url}")
            except sqlite3.Error as db_err:
                logger.error(f"Erro ao salvar no DB para {original_filename}: {db_err}")
                failed_uploads.append({"filename": original_filename, "reason": f"Erro no DB: {db_err}"})

        except Exception as e:
            logger.error(f"Erro durante o upload de {original_filename}: {e}")
            failed_uploads.append({"filename": original_filename, "reason": f"Erro no upload: {str(e)}"})

    conn.close()
    
    summary = {
        "message": f"Processo de upload concluído. {uploaded_count} imagens processadas com sucesso.",
        "uploaded_successfully": uploaded_count,
        "failed_uploads_count": len(failed_uploads),
        "failures": failed_uploads
    }
    logger.info(summary)
    return summary

@app.post("/admin/upload-local-garments")
async def trigger_local_garment_upload():
    """
    [ADMIN] Endpoint para fazer upload das imagens da pasta /images para o Supabase.
    Use com cautela e preferencialmente apenas uma vez.
    NAO USAR A MENOS QUE PRECISE FAZER UPLOAD NOVAMENTE PARA O SUPABASE
    """
    
    logger.info("Iniciando upload de imagens locais de vestuário para o Supabase...")
    result = await upload_local_garment_images_to_supabase(num_images=1000) # Ajuste num_images conforme necessário
    return result
    
@app.get("/clothes", response_model=List[ClothItemDetail])
async def get_random_clothes():
    if df_clothes_dataset is None:
        raise HTTPException(status_code=503, detail="Dataset não carregado.")
    if df_clothes_dataset.empty:
        return []

    sample_df = df_clothes_dataset.sample(min(10, len(df_clothes_dataset)))
    
    tasks_openai = []
    # Coletar IDs para buscar URLs do Supabase de uma vez, se possível, ou conectar uma vez
    
    for idx, row in sample_df.iterrows():
        text_value = row.get("text", "")
        if isinstance(text_value, bytes):
            text_value = text_value.decode("utf-8", "replace")
        
        # O 'id' da roupa no seu dataset. Usaremos para a OpenAI e para buscar a URL no Supabase.
        current_garment_id = str(idx) 

        # Adiciona a task para a OpenAI
        tasks_openai.append(
            gen_title_description(text_value, current_garment_id) # A função gen_title_description já existe no seu código
        )

    # Executa todas as chamadas à OpenAI em paralelo
    openai_results = await asyncio.gather(*tasks_openai, return_exceptions=True)

    results = []
    # Abre a conexão com o SQLite uma vez para buscar as URLs
    conn_sqlite = None
    try:
        conn_sqlite = sqlite3.connect(DATABASE_URL)
        cursor_sqlite = conn_sqlite.cursor()

        # Itera sobre os resultados do dataset e os resultados da OpenAI
        for (idx, row), openai_data in zip(sample_df.iterrows(), openai_results):
            title = "Título não gerado"
            description  = "Descrição não gerada"
            
            if isinstance(openai_data, Exception):
                logger.error(f"Erro na chamada OpenAI para o item com idx={idx}: {openai_data}")
            elif openai_data: # Verifica se openai_data não é None ou uma exceção já tratada
                title = openai_data.get("title", title)
                description = openai_data.get("description", description)

            garment_id_str = str(idx) 
            image_url = "/static_images/default_image_placeholder.png" # URL Padrão

            cursor_sqlite.execute(
                "SELECT supabase_url FROM garment_images WHERE garment_dataset_id = ?",
                (garment_id_str,)
            )
            url_row = cursor_sqlite.fetchone()

            if url_row and url_row[0]:
                image_url = url_row[0]
            else:
                logger.warning(f"URL do Supabase NÃO encontrada para garment_id {garment_id_str}. Usando placeholder.")


            original_text_value = row.get("text", "")
            if isinstance(original_text_value, bytes):
                original_text_value = original_text_value.decode("utf-8", "replace")

            results.append(ClothItemDetail(
                id=garment_id_str,
                image_url=image_url,
                title=title,
                description=description,
                original_text=original_text_value
            ))
    
    except sqlite3.Error as e:
        logger.error(f"Erro de banco de dados ao buscar URLs de imagem: {e}")
        if not results:
             raise HTTPException(status_code=500, detail="Erro ao acessar dados das imagens.")

    finally:
        if conn_sqlite:
            conn_sqlite.close()
            
    return results

@app.post("/clothes")
def save_image_ids(image_data: ImageIDs):
    """Recebe uma lista de IDs de imagem, classifica suas features usando OpenAI,
    determina o perfil de moda usando RandomForest e salva no banco de dados."""
    if not all([pipeline_model, cluster_names_map, rf_model]):
        raise HTTPException(status_code=503, detail="Modelos de classificação não estão prontos. Tente novamente mais tarde.")

    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    saved_count = 0
    errors = []

    for raw_id_str in image_data.ids:
        try:
            idx = int(raw_id_str)
        except ValueError:
            errors.append(f"ID inválido (não numérico): {raw_id_str}")
            continue

        if df_clothes_dataset is None or 'text' not in df_clothes_dataset.columns:
            errors.append(f"Dataset de roupas não carregado ou coluna 'text' ausente para ID {idx}.")
            continue
        if idx not in df_clothes_dataset.index:
            errors.append(f"ID {idx} não encontrado no dataset de roupas.")
            continue

        text_value = df_clothes_dataset.loc[idx, 'text']
        if isinstance(text_value, bytes):
            text_value = text_value.decode('utf-8', errors='replace')


        try:
            openai_response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": (
                        "Você é um assistente que analisa descrições de roupas e atribui pontuações de 1 a 10 para 5 características de estilo: "
                        "'cores_vivas', 'versatilidade', 'conforto', 'formalidade', 'estampas'. "
                        "Responda estritamente no formato JSON especificado."
                    )},
                    {"role": "user", "content": f"Analise esta descrição de roupa: '{text_value}'"}
                ],
                response_format={
                    "type": "json_object", 
                }
            )
            content_str = openai_response.choices[0].message.content
            features = ClothingFeatures.model_validate_json(content_str)

        except Exception as e:
            logger.error(f"Erro na chamada OpenAI para ID {idx} ('{text_value}'): {e}")
            errors.append(f"Falha na OpenAI para ID {idx}: {str(e)}")
            continue

        try:
            features_array = np.array([[
                features.cores_vivas,
                features.versatilidade,
                features.conforto,
                features.formalidade,
                features.estampas
            ]])
            
            predicted_cluster_id = int(rf_model.predict(features_array)[0])
            predicted_profile_name = cluster_names_map.get(predicted_cluster_id, "Desconhecido")

        except Exception as e:
            logger.error(f"Erro ao classificar perfil para ID {idx} com RandomForest: {e}")
            errors.append(f"Erro na classificação do perfil para ID {idx}: {str(e)}")
            predicted_cluster_id = None 
            predicted_profile_name = "Erro na Classificação"


        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO saved_ids
                  (image_id, text, cores_vivas, versatilidade, conforto, formalidade, estampas, cluster_id, cluster_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(idx), text_value,
                    features.cores_vivas, features.versatilidade, features.conforto,
                    features.formalidade, features.estampas,
                    predicted_cluster_id, predicted_profile_name
                )
            )
            if cursor.rowcount > 0:
                saved_count += 1
        except sqlite3.Error as db_e:
            logger.error(f"Erro no DB ao salvar ID {idx}: {db_e}")
            errors.append(f"Erro no DB para ID {idx}: {db_e}")

    conn.commit()
    conn.close()

    if errors:
        return {
            "message": f"{saved_count} de {len(image_data.ids)} IDs processados. Alguns com sucesso.",
            "saved_successfully": saved_count,
            "errors_occurred": len(errors),
            "error_details": errors
        }

    return {"message": f"{saved_count} de {len(image_data.ids)} IDs salvos com sucesso."}


@app.post("/classify_profile", response_model=ProfileResponse)
def classify_user_profile(user_data: UserProfileInput):
    """
    Classifica o perfil do usuário com base nas features fornecidas (escala 1-10),
    usando o modelo RandomForest treinado.
    """
    if not all([pipeline_model, cluster_names_map, rf_model]):
        raise HTTPException(status_code=503, detail="Modelos de classificação não estão prontos.")

    features_input = [
        user_data.features.cores_vivas,
        user_data.features.versatilidade,
        user_data.features.conforto,
        user_data.features.formalidade,
        user_data.features.estampas
    ]
    features_array = np.array([features_input])

    try:
        cluster_id = int(rf_model.predict(features_array)[0])
        profile_name = cluster_names_map.get(cluster_id, "Desconhecido")
        description = PROFILE_DESCRIPTIONS.get(profile_name, "Descrição não disponível.")
        
        return ProfileResponse(
            cluster_id=cluster_id,
            profile_name=profile_name,
            description=description
        )
    except Exception as e:
        logger.error(f"Erro ao classificar perfil do usuário: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao classificar perfil: {str(e)}"
        )
    
@app.get("/user_profile", response_model=ProfileResponse)
def get_user_profile_from_saved_clothes():
    """
    Calcula o perfil médio do usuário com base nas roupas salvas e o classifica
    usando o modelo RandomForest.
    """
    if not all([pipeline_model, cluster_names_map, rf_model]):
        raise HTTPException(status_code=503, detail="Modelos de classificação não estão prontos.")
    
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM saved_ids")
    count = cursor.fetchone()[0]
    
    if count == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Nenhuma roupa salva para calcular o perfil do usuário.")
    
    cursor.execute("""
        SELECT 
            AVG(cores_vivas) as avg_cores_vivas,
            AVG(versatilidade) as avg_versatilidade,
            AVG(conforto) as avg_conforto,
            AVG(formalidade) as avg_formalidade,
            AVG(estampas) as avg_estampas
        FROM saved_ids
        WHERE cores_vivas IS NOT NULL  -- Considerar apenas registros com features válidas
    """)
    avg_features_tuple = cursor.fetchone()
    conn.close()

    if not all(f is not None for f in avg_features_tuple):
        raise HTTPException(status_code=404, detail="Não há dados de features suficientes para calcular o perfil.")

    avg_features_array = np.array([[round(f) for f in avg_features_tuple]])
    
    try:
        cluster_id = int(rf_model.predict(avg_features_array)[0])
        profile_name = cluster_names_map.get(cluster_id, "Desconhecido")
        description = PROFILE_DESCRIPTIONS.get(profile_name, "Descrição não disponível.")
        
        return ProfileResponse(
            cluster_id=cluster_id,
            profile_name=profile_name,
            description=description
        )
    except Exception as e:
        logger.error(f"Erro ao obter perfil do usuário agregado: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar perfil do usuário: {str(e)}"
        )

def generate_unique_filename(original_filename: str) -> str:
    """Gera um nome único para o arquivo baseado em timestamp e UUID"""
    file_extension = os.path.splitext(original_filename)[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}{file_extension}"

# Função para validar tipo de arquivo de imagem
def validate_image_file(file: UploadFile) -> bool:
    """Valida se o arquivo é uma imagem válida"""
    allowed_types = [
        "image/jpeg", "image/jpg", "image/png", 
        "image/gif", "image/webp", "image/bmp"
    ]
    return file.content_type in allowed_types

@app.post("/upload-images", response_model=UploadResponse)
async def upload_images_to_supabase(files: List[UploadFile] = File(...)):
    """
    Endpoint para fazer upload de múltiplas imagens para o Supabase Storage.
    
    Args:
        files: Lista de arquivos de imagem para upload
        
    Returns:
        UploadResponse com detalhes dos arquivos enviados e falhas
    """
    uploaded_files = []
    failed_files = []
    
    # Conecta ao banco local para registrar uploads
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    for file in files:
        try:
            # Valida se é um arquivo de imagem
            if not validate_image_file(file):
                failed_files.append({
                    "filename": file.filename,
                    "error": f"Tipo de arquivo não suportado: {file.content_type}"
                })
                continue
            
            # Lê o conteúdo do arquivo
            file_content = await file.read()
            file_size = len(file_content)
            
            # Gera nome único para o arquivo
            unique_filename = generate_unique_filename(file.filename)
            supabase_path = f"uploads/{unique_filename}"
            
            # Faz upload para o Supabase Storage
            try:
                upload_result = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(
                    path=supabase_path,
                    file=file_content,
                    file_options={
                        "content-type": file.content_type,
                        "upsert": False  # Não sobrescreve arquivos existentes
                    }
                )
                
                # Gera URL pública do arquivo
                public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(supabase_path)
                
                # Salva informações no banco local
                cursor.execute(
                    """
                    INSERT INTO uploaded_images 
                    (filename, supabase_path, public_url, file_size, content_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (file.filename, supabase_path, public_url, file_size, file.content_type)
                )
                
                uploaded_files.append({
                    "original_filename": file.filename,
                    "supabase_path": supabase_path,
                    "public_url": public_url,
                    "file_size": f"{file_size} bytes"
                })
                
                logger.info(f"Upload bem-sucedido: {file.filename} -> {supabase_path}")
                
            except Exception as supabase_error:
                failed_files.append({
                    "filename": file.filename,
                    "error": f"Erro no Supabase: {str(supabase_error)}"
                })
                logger.error(f"Erro no upload para Supabase - {file.filename}: {supabase_error}")
                
        except Exception as e:
            failed_files.append({
                "filename": file.filename if file.filename else "arquivo_sem_nome",
                "error": f"Erro geral: {str(e)}"
            })
            logger.error(f"Erro geral no upload - {file.filename}: {e}")
    
    # Confirma transações no banco local
    conn.commit()
    conn.close()
    
    # Prepara resposta
    total_uploaded = len(uploaded_files)
    total_failed = len(failed_files)
    
    message = f"Upload concluído: {total_uploaded} arquivo(s) enviado(s) com sucesso"
    if total_failed > 0:
        message += f", {total_failed} arquivo(s) falharam"
    
    return UploadResponse(
        message=message,
        uploaded_files=uploaded_files,
        failed_files=failed_files
    )

@app.get("/uploaded-images")
def get_uploaded_images():
    """
    Retorna lista de todas as imagens que foram enviadas para o Supabase.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, filename, supabase_path, public_url, upload_date, file_size, content_type
        FROM uploaded_images 
        ORDER BY upload_date DESC
    """)
    
    images = []
    for row in cursor.fetchall():
        images.append({
            "id": row[0],
            "filename": row[1],
            "supabase_path": row[2],
            "public_url": row[3],
            "upload_date": row[4],
            "file_size": row[5],
            "content_type": row[6]
        })
    
    conn.close()
    
    return {
        "total_images": len(images),
        "images": images
    }

@app.delete("/uploaded-images/{image_id}")
def delete_uploaded_image(image_id: int):
    """
    Remove uma imagem do Supabase Storage e do banco local.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Busca informações da imagem
    cursor.execute(
        "SELECT supabase_path, filename FROM uploaded_images WHERE id = ?",
        (image_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    supabase_path, filename = result
    
    try:
        # Remove do Supabase Storage
        supabase.storage.from_(SUPABASE_BUCKET_NAME).remove([supabase_path])
        
        # Remove do banco local
        cursor.execute("DELETE FROM uploaded_images WHERE id = ?", (image_id,))
        conn.commit()
        conn.close()
        
        return {"message": f"Imagem '{filename}' removida com sucesso"}
        
    except Exception as e:
        conn.close()
        logger.error(f"Erro ao remover imagem {image_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao remover imagem: {str(e)}"
        )