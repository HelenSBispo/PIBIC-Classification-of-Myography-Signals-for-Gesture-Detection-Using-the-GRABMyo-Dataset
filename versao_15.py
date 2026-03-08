import os
import time
import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import scipy.signal
from scipy.stats import mode as scipy_mode, entropy as scipy_entropy
from tqdm.auto import tqdm as tqdm_notebook # Or just from tqdm import tqdm as tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, log_loss,
                             mean_squared_error, mean_absolute_error, r2_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier

USAR_DADOS_BRUTOS = True
PORCENTAGEM_DADOS_USAR_TOTAL = 0.2

# NOVO: Caminho para carregar um modelo previamente salvo.
# Se vazio ou None, o script procederá com o GridSearchCV e treinamento.
# Certifique-se de que este caminho aponta para o arquivo .keras/.h5 real do modelo.
# Exemplo: CAMINHO_MODELO_CARREGAR = 'D:/Dowloads/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0/melhor_modelo_cnn_emg_raw_Trueperc20.keras'
CAMINHO_MODELO_CARREGAR = 'D:/Dowloads/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0/melhor_modelo_cnn_emg_raw_Trueperc20.keras'


print(f"INFO: USAR_DADOS_BRUTOS: {USAR_DADOS_BRUTOS}")
print(f"INFO: PORCENTAGEM_DADOS_USAR_TOTAL: {PORCENTAGEM_DADOS_USAR_TOTAL*100}%")
if CAMINHO_MODELO_CARREGAR:
    print(f"INFO: CAMINHO_MODELO_CARREGAR definido: '{CAMINHO_MODELO_CARREGAR}'")
    print("ATENÇÃO: O script TENTARÁ CARREGAR ESTE MODELO, PULANDO O TREINAMENTO E O GridSearchCV.")
else:
    print("INFO: CAMINHO_MODELO_CARREGAR não definido. O script realizará GridSearchCV e treinamento.")

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive Montado!")
    CAMINHO_BASE_DADOS = '/content/drive/MyDrive/GrabMyo/physionet.org/files/grabmyo/1.1.0'
    if not os.path.exists(CAMINHO_BASE_DADOS):
        raise FileNotFoundError(f"ERRO: Caminho base de dados não encontrado: {CAMINHO_BASE_DADOS}")
    else:
        print(f"Caminho base de dados OK: {CAMINHO_BASE_DADOS}")
except ModuleNotFoundError:
    print("INFO: Google Drive não disponível (não estamos no Colab?).")
    # Ajuste o CAMINHO_BASE_DADOS para o seu caminho local
    CAMINHO_BASE_DADOS = 'D:/Dowloads/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0'
    if not os.path.exists(CAMINHO_BASE_DADOS):
        print(f"AVISO URGENTE: Caminho base de dados LOCAL '{CAMINHO_BASE_DADOS}' não encontrado! Verifique e ajuste.")
    else:
        print(f"Caminho base de dados LOCAL OK: {CAMINHO_BASE_DADOS}")
except Exception as e:
    print(f"Falha ao montar Drive ou verificar caminho: {e}")
    raise

SESSOES_A_CARREGAR = ['Session1'] # ['Session1', 'Session2']

FREQ_AMOSTRAGEM = 2048
DURACAO_SEGMENTO_MS = 500
COMPRIMENTO_SEGMENTO = int(FREQ_AMOSTRAGEM * (DURACAO_SEGMENTO_MS / 1000))

FREQ_CORTE_INFERIOR = 20.0
FREQ_CORTE_SUPERIOR = 500.0
ORDEM_FILTRO = 4

CANAIS_TOTAIS_ESPERADOS = 32
INDICES_CANAIS_EMG = list(range(12))
INDICE_CANAL_LABEL = 31
NUM_CANAIS_EMG = len(INDICES_CANAIS_EMG)

DICIONARIO_CLASSES_GESTOS = {
    0: 'Repouso', 1: 'Cilíndrico', 2: 'Ponta dos Dedos', 3: 'Palmar (Pinça Normal)',
    4: 'Lateral (Pinça Chave)', 5: 'Gancho', 6: 'Esférico'
}

TAMANHO_BUFFER_EMBARALHAR = 1024 * 2
TAMANHO_LOTE = 32
PROPORCAO_TESTE = 0.2
ESTADO_ALEATORIO = 42

EPOCAS_GRID_SEARCH = 5
EPOCAS_FINAIS = 10
PACIENCIA_EARLY_STOPPING = 5
PACIENCIA_REDUCE_LR = 3

FUNCOES_ATIVACAO_GRID = ['relu', 'elu']
DICIONARIO_OTIMIZADORES = {
    "Adam": optimizers.Adam, "AdamW": optimizers.AdamW,
    "RMSprop": optimizers.RMSprop
}
OPCOES_OTIMIZADORES_GRID = list(DICIONARIO_OTIMIZADORES.keys())
TAXAS_APRENDIZADO_GRID = [1e-4, 5e-4]
DROPOUT_RATES_GRID = [0.25, 0.4]
FILTROS_CONV1_GRID = [64, 128]

# O nome do arquivo salvo pode depender dos parâmetros.
# Se você está carregando um modelo, este nome é para onde ele seria salvo SE fosse treinado.
NOME_ARQUIVO_MODELO_SALVO = f'melhor_modelo_cnn_emg_raw_{USAR_DADOS_BRUTOS}perc{int(PORCENTAGEM_DADOS_USAR_TOTAL*100)}.keras' # Extensão .keras é o formato nativo do TF

print("-" * 50)
print("Constantes e Configurações Definidas:")
print(f"  Freq Amostragem: {FREQ_AMOSTRAGEM} Hz")
print(f"  Comprimento Segmento: {COMPRIMENTO_SEGMENTO} amostras ({DURACAO_SEGMENTO_MS} ms)")
print(f"  Canais EMG: {NUM_CANAIS_EMG} (Índices {INDICES_CANAIS_EMG})")
print(f"  Canal Label: {INDICE_CANAL_LABEL}")
print(f"  Usar Dados Brutos (sem filtro/norm): {USAR_DADOS_BRUTOS}")
print(f"  Porcentagem de Dados a Utilizar: {PORCENTAGEM_DADOS_USAR_TOTAL*100}%")
if not CAMINHO_MODELO_CARREGAR:
    print(f"  GridSearchCV - Épocas: {EPOCAS_GRID_SEARCH}")
    print(f"  GridSearchCV - Ativações: {FUNCOES_ATIVACAO_GRID}")
    print(f"  GridSearchCV - Otimizadores: {OPCOES_OTIMIZADORES_GRID}")
    print(f"  GridSearchCV - Taxas Aprendizado: {TAXAS_APRENDIZADO_GRID}")
    print(f"  GridSearchCV - Dropouts: {DROPOUT_RATES_GRID}")
    print(f"  GridSearchCV - Filtros C1: {FILTROS_CONV1_GRID}")
print(f"  Modelo será salvo como: '{NOME_ARQUIVO_MODELO_SALVO}' (se o treinamento for realizado)")
if CAMINHO_MODELO_CARREGAR:
    print(f"  Tentando carregar modelo de: '{CAMINHO_MODELO_CARREGAR}'")
print("-" * 50)

def filtrar_passa_banda_butterworth(dados, fi=FREQ_CORTE_INFERIOR, fs=FREQ_CORTE_SUPERIOR, fq=FREQ_AMOSTRAGEM, ord=ORDEM_FILTRO):
    nyq = 0.5 * fq
    low = fi / nyq
    high = fs / nyq
    high = min(high, 0.99999)
    low = max(low, 0.00001)
    if high <= low:
        print(f"AVISO Filtro: Freq. corte superior ({high*nyq} Hz) <= Freq. corte inferior ({low*nyq} Hz). Retornando dados originais.")
        return dados
    try:
        sos = scipy.signal.butter(ord, [low, high], btype='band', output='sos')
        dados_filtrados = scipy.signal.sosfiltfilt(sos, dados, axis=0)
        return dados_filtrados
    except ValueError as e:
        print(f"ERRO no filtro Butterworth: {e}. Retornando dados originais.")
        return dados

def normalizar_sinal(sinal):
    media = np.mean(sinal, axis=0)
    dp = np.std(sinal, axis=0)
    dp[dp < 1e-7] = 1.0
    return (sinal - media) / dp

def calcular_sens_espec(matriz_confusao):
    num_classes = matriz_confusao.shape[0]
    sensibilidades = np.zeros(num_classes)
    especificidades = np.zeros(num_classes)
    for k in range(num_classes):
        TP = matriz_confusao[k, k]
        FN = np.sum(matriz_confusao[k, :]) - TP
        FP = np.sum(matriz_confusao[:, k]) - TP
        TN = np.sum(matriz_confusao) - (TP + FN + FP)
        sensibilidades[k] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        especificidades[k] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return sensibilidades, especificidades

def calcular_entropia_shannon(prob_predicoes):
    prob_predicoes = np.clip(prob_predicoes, 1e-10, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropias = scipy_entropy(prob_predicoes, axis=1, base=2)
    return np.mean(entropias[np.isfinite(entropias)])

print("INFO: Funções auxiliares de pré-processamento e métricas definidas.")

def gerador_dados(caminho_base, sessoes, comp_segmento, freq_amostragem,
                  canais_totais, indices_emg, indice_label, scan=False, usar_brutos_local=USAR_DADOS_BRUTOS):
    if not scan:
        print(f"\nINFO Gerador: Iniciando geração de dados... (Usar Brutos: {usar_brutos_local})")

    contador_segmentos_validos = 0
    num_canais_emg = len(indices_emg)
    primeira_msg_tipo_dado = True

    for nome_sessao in sessoes:
        caminho_sessao = os.path.join(caminho_base, nome_sessao)
        if not os.path.isdir(caminho_sessao):
            if not scan: print(f"AVISO Gerador: Diretório da sessão '{nome_sessao}' não encontrado. Pulando.")
            continue

        if not scan: print(f"INFO Gerador: Processando Sessão '{nome_sessao}'...")
        arquivos_dat = []
        for root, _, files in os.walk(caminho_sessao):
            for file_name in files:
                if file_name.endswith('.dat'):
                    arquivos_dat.append(os.path.join(root, file_name.replace('.dat', '')))
        arquivos_dat.sort()

        for caminho_registro_sem_ext in arquivos_dat:
            try:
                registro = wfdb.rdrecord(caminho_registro_sem_ext, warn_empty=True)
                if registro.fs != freq_amostragem:
                    if not scan: print(f"AVISO Gerador: Freq. amostragem incompatível em {caminho_registro_sem_ext}. Esperado {freq_amostragem}, encontrado {registro.fs}. Pulando.")
                    continue
                if registro.p_signal is None:
                    if not scan: print(f"AVISO Gerador: Sinal nulo em {caminho_registro_sem_ext}. Pulando.")
                    continue
                if registro.sig_len < comp_segmento:
                    if not scan: print(f"AVISO Gerador: Sinal muito curto em {caminho_registro_sem_ext}. Pulando.")
                    continue
                if registro.n_sig != canais_totais:
                    if not scan: print(f"AVISO Gerador: Número de canais incompatível em {caminho_registro_sem_ext}. Pulando.")
                    continue

                sinais = registro.p_signal
                if np.isnan(sinais).any():
                    if not scan: print(f"AVISO Gerador: NaN encontrado nos sinais de {caminho_registro_sem_ext}. Pulando.")
                    continue

                sinais_emg_originais = sinais[:, indices_emg]
                labels_tempo = sinais[:, indice_label]

                if usar_brutos_local:
                    sinais_emg_processados = sinais_emg_originais
                    if not scan and primeira_msg_tipo_dado:
                        print("INFO Gerador: Utilizando dados EMG brutos (sem filtro/normalização).")
                        primeira_msg_tipo_dado = False
                else:
                    sinais_emg_filtrados = filtrar_passa_banda_butterworth(sinais_emg_originais, fq=registro.fs)
                    sinais_emg_processados = normalizar_sinal(sinais_emg_filtrados)
                    if not scan and primeira_msg_tipo_dado:
                        print("INFO Gerador: Utilizando dados EMG filtrados e normalizados.")
                        primeira_msg_tipo_dado = False

                num_segmentos_no_arquivo = sinais_emg_processados.shape[0] // comp_segmento
                for i in range(num_segmentos_no_arquivo):
                    idx_inicio = i * comp_segmento
                    idx_fim = idx_inicio + comp_segmento
                    segmento_emg = sinais_emg_processados[idx_inicio:idx_fim, :]
                    segmento_labels = labels_tempo[idx_inicio:idx_fim]

                    if segmento_emg.shape == (comp_segmento, num_canais_emg):
                        mode_result = scipy_mode(segmento_labels.astype(int), keepdims=False)
                        label_segmento = mode_result.mode
                        if hasattr(label_segmento, "__len__"):
                            if len(label_segmento) == 0: continue
                            label_segmento = label_segmento[0]

                        if np.isnan(label_segmento): continue
                        contador_segmentos_validos += 1
                        if not scan:
                            yield segmento_emg.astype(np.float32), np.int32(label_segmento)
            except FileNotFoundError:
                if not scan: print(f"AVISO Gerador: Arquivo .hea ou .dat não encontrado para '{caminho_registro_sem_ext}'. Pulando.")
            except ValueError as ve:
                if not scan: print(f"AVISO Gerador: Erro de valor ao processar '{caminho_registro_sem_ext}': {ve}. Pulando.")
            except Exception as e:
                if not scan: print(f"ERRO Gerador: Erro inesperado ao processar '{caminho_registro_sem_ext}': {e}. Pulando.")
            finally:
                if 'registro' in locals(): del registro
                if 'sinais' in locals(): del sinais
                gc.collect()
    if not scan:
        print(f"INFO Gerador: Geração concluída. Total de {contador_segmentos_validos} segmentos produzidos.")
    if contador_segmentos_validos == 0 and not scan:
        print("\nERRO CRÍTICO: O gerador de dados não produziu nenhum segmento válido. Verifique os caminhos, parâmetros e dados.")

print("\n--- Contagem de Segmentos e Preparação de Labels ---")
lista_labels_originais = []
total_segmentos_estimado_real = 0
gerador_para_contagem = gerador_dados(CAMINHO_BASE_DADOS, SESSOES_A_CARREGAR, COMPRIMENTO_SEGMENTO, FREQ_AMOSTRAGEM,
                                      CANAIS_TOTAIS_ESPERADOS, INDICES_CANAIS_EMG, INDICE_CANAL_LABEL,
                                      scan=False, usar_brutos_local=USAR_DADOS_BRUTOS)
print("INFO: Coletando labels reais para codificação (respeitando config. USAR_DADOS_BRUTOS)...")
iterador_contagem = iter(tqdm_notebook(gerador_para_contagem, desc="Coletando Labels Reais"))
try:
    for _, label in iterador_contagem:
        lista_labels_originais.append(label)
        total_segmentos_estimado_real += 1
except Exception as e:
    print(f"Erro durante a coleta de labels: {e}")
finally:
    if hasattr(gerador_para_contagem, 'close'):
        gerador_para_contagem.close()

if total_segmentos_estimado_real == 0:
    raise ValueError("ERRO CRÍTICO: Nenhum segmento válido foi encontrado durante a contagem. Verifique as configurações e os dados.")
print(f"INFO: Contagem real concluída. Total de Segmentos Válidos Encontrados: {total_segmentos_estimado_real}")

y_labels_originais_todos = np.array(lista_labels_originais, dtype=np.int32)
del lista_labels_originais
gc.collect()

print("\nINFO: Codificando labels...")
codificador_label = LabelEncoder()
codificador_label.fit(y_labels_originais_todos)
ids_classes_encontradas = codificador_label.classes_
NUM_CLASSES = len(ids_classes_encontradas)
print(f"INFO: Número final de classes detectadas: {NUM_CLASSES} (IDs Originais dos Gestos: {ids_classes_encontradas})")

nomes_classes_ordenados = [DICIONARIO_CLASSES_GESTOS.get(int(id_original), f"ID Desconhecido {id_original}")
                           for id_original in ids_classes_encontradas]
print(f"INFO: Nomes das Classes (na ordem codificada 0..{NUM_CLASSES-1}): {nomes_classes_ordenados}")

print("\nCriação e Preparação dos Datasets tf.data")
def fabrica_gerador_principal():
    return gerador_dados(CAMINHO_BASE_DADOS, SESSOES_A_CARREGAR, COMPRIMENTO_SEGMENTO, FREQ_AMOSTRAGEM,
                         CANAIS_TOTAIS_ESPERADOS, INDICES_CANAIS_EMG, INDICE_CANAL_LABEL,
                         scan=False, usar_brutos_local=USAR_DADOS_BRUTOS)

assinatura_saida = (
    tf.TensorSpec(shape=(COMPRIMENTO_SEGMENTO, NUM_CANAIS_EMG), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

print("INFO: Criando Dataset completo a partir do gerador...")
dataset_completo = tf.data.Dataset.from_generator(
    fabrica_gerador_principal,
    output_signature=assinatura_saida
)

if PORCENTAGEM_DADOS_USAR_TOTAL < 1.0:
    total_segmentos_para_uso = int(total_segmentos_estimado_real * PORCENTAGEM_DADOS_USAR_TOTAL)
    if total_segmentos_para_uso == 0 and total_segmentos_estimado_real > 0:
        total_segmentos_para_uso = 1
        print(f"AVISO: {PORCENTAGEM_DADOS_USAR_TOTAL*100}% resultou em 0 segmentos, usando 1 segmento para evitar erro.")

    print(f"INFO: Utilizando {PORCENTAGEM_DADOS_USAR_TOTAL*100}% dos dados: {total_segmentos_para_uso} segmentos.")
    dataset_completo = dataset_completo.take(total_segmentos_para_uso)
    segmentos_efetivos_para_split = total_segmentos_para_uso
    tamanho_buffer_embaralhar_efetivo = min(TAMANHO_BUFFER_EMBARALHAR, max(1, segmentos_efetivos_para_split))
else:
    segmentos_efetivos_para_split = total_segmentos_estimado_real
    tamanho_buffer_embaralhar_efetivo = TAMANHO_BUFFER_EMBARALHAR
    print(f"INFO: Utilizando 100% dos dados: {segmentos_efetivos_para_split} segmentos.")

if segmentos_efetivos_para_split == 0:
    raise ValueError("ERRO CRÍTICO: Nenhum segmento disponível para treino/teste após aplicar porcentagem. Verifique os dados e a configuração.")

print(f"INFO: Embaralhando Dataset (buffer={tamanho_buffer_embaralhar_efetivo})...")
dataset_completo = dataset_completo.shuffle(tamanho_buffer_embaralhar_efetivo, seed=ESTADO_ALEATORIO, reshuffle_each_iteration=True)

print(f"INFO: Dividindo Dataset em Treino/Teste ({100 - PROPORCAO_TESTE * 100:.0f}% / {PROPORCAO_TESTE * 100:.0f}%)...")
tamanho_teste = int(segmentos_efetivos_para_split * PROPORCAO_TESTE)
if tamanho_teste == 0 and segmentos_efetivos_para_split > 0:
    tamanho_teste = 1
tamanho_treino = segmentos_efetivos_para_split - tamanho_teste
if tamanho_treino == 0 and segmentos_efetivos_para_split > tamanho_teste:
    tamanho_treino = 1
    tamanho_teste = segmentos_efetivos_para_split - 1
    if tamanho_teste < 0: tamanho_teste = 0

dataset_teste_raw = dataset_completo.take(tamanho_teste)
dataset_treino_raw = dataset_completo.skip(tamanho_teste)

print(f"INFO: Estimativa REAL - {tamanho_treino} amostras de treino, {tamanho_teste} amostras de teste.")
if tamanho_treino == 0 or tamanho_teste == 0:
    print(f"AVISO: Dataset de treino ({tamanho_treino}) ou teste ({tamanho_teste}) possui 0 amostras. Isso pode causar erros. Verifique PORCENTAGEM_DADOS_USAR_TOTAL e PROPORCAO_TESTE.")
    if tamanho_treino == 0 and tamanho_teste == 0 and segmentos_efetivos_para_split > 0:
        raise ValueError("ERRO CRÍTICO: Treino e Teste com 0 amostras, mas havia segmentos. Lógica de divisão falhou.")

original_class_ids_tensor = tf.constant(codificador_label.classes_, dtype=tf.int64)
encoded_class_ids_tensor = tf.range(NUM_CLASSES, dtype=tf.int64)

label_lookup_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=original_class_ids_tensor,
        values=encoded_class_ids_tensor
    ),
    default_value=-1
)

def preparar_lote_para_treino_keras(segmento, rotulo_original_tensor):

    rotulo_codificado_tensor = label_lookup_table.lookup(tf.cast(rotulo_original_tensor, tf.int64))
    rotulo_one_hot = tf.one_hot(rotulo_codificado_tensor, depth=NUM_CLASSES, dtype=tf.float32)
    return segmento, rotulo_one_hot

def preparar_lote_para_avaliacao_scikeras(segmento, rotulo_original_tensor):
    rotulo_codificado_tensor = label_lookup_table.lookup(tf.cast(rotulo_original_tensor, tf.int64))
    return segmento, tf.cast(rotulo_codificado_tensor, tf.int32)

def preparar_lote_para_metricas_finais(segmento, rotulo_original_tensor):
    rotulo_codificado_tensor = label_lookup_table.lookup(tf.cast(rotulo_original_tensor, tf.int64))
    return segmento, tf.cast(rotulo_codificado_tensor, tf.int32)

print("INFO: Otimizando Datasets de Treino e Teste (batch, map, prefetch)...")
dataset_treino = None
# Apenas crie o dataset de treino se você for realmente treinar um modelo
if tamanho_treino > 0 and not CAMINHO_MODELO_CARREGAR:
    dataset_treino = dataset_treino_raw.map(preparar_lote_para_treino_keras, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_treino = dataset_treino.batch(TAMANHO_LOTE)
    shuffle_buffer_batched = max(1, tamanho_buffer_embaralhar_efetivo // TAMANHO_LOTE)
    dataset_treino = dataset_treino.shuffle(shuffle_buffer_batched, seed=ESTADO_ALEATORIO, reshuffle_each_iteration=True)
    dataset_treino = dataset_treino.prefetch(tf.data.AUTOTUNE)
    print(f"INFO: Estrutura do Lote de Treino: {dataset_treino.element_spec}")
else:
    print("AVISO: Dataset de Treino não será processado (vazio ou carregando modelo existente).")

dataset_teste_eval = None
if tamanho_teste > 0:
    dataset_teste_eval = dataset_teste_raw.map(preparar_lote_para_avaliacao_scikeras, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_teste_eval = dataset_teste_eval.batch(TAMANHO_LOTE)
    dataset_teste_eval = dataset_teste_eval.prefetch(tf.data.AUTOTUNE)
    print(f"INFO: Estrutura do Lote de Teste (Avaliação GridSearchCV): {dataset_teste_eval.element_spec}")
else:
    print("AVISO: Dataset de Teste (para avaliação GridSearchCV) vazio. Não será processado.")

dataset_teste_metrics = None
if tamanho_teste > 0:
    dataset_teste_metrics = dataset_teste_raw.map(preparar_lote_para_metricas_finais, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_teste_metrics = dataset_teste_metrics.batch(TAMANHO_LOTE)
    dataset_teste_metrics = dataset_teste_metrics.prefetch(tf.data.AUTOTUNE)
    print(f"INFO: Estrutura do Lote de Teste (Métricas Finais): {dataset_teste_metrics.element_spec}")
else:
    print("AVISO: Dataset de Teste (para métricas finais) vazio. Não será processado.")

if 'dataset_completo' in locals(): del dataset_completo
if 'dataset_treino_raw' in locals(): del dataset_treino_raw
if 'dataset_teste_raw' in locals(): del dataset_teste_raw
if 'y_labels_originais_todos' in locals(): del y_labels_originais_todos
gc.collect()

if dataset_treino and not CAMINHO_MODELO_CARREGAR: # Só plota se for treinar um novo modelo
    try:
        lote_exemplo_tensor = next(iter(dataset_treino.take(1)))
        segmentos_exemplo, labels_exemplo_one_hot = lote_exemplo_tensor

        num_exemplos_plotar = min(3, segmentos_exemplo.shape[0])
        num_canais_plotar = min(NUM_CANAIS_EMG, 3)
        eixo_tempo = np.linspace(0, DURACAO_SEGMENTO_MS, COMPRIMENTO_SEGMENTO)

        plt.figure(figsize=(16, num_exemplos_plotar * 2.8))
        titulo_sufixo = "(Dados Brutos)" if USAR_DADOS_BRUTOS else "(Dados Pré-processados)"
        plt.suptitle(f'Exemplos de Segmentos EMG {titulo_sufixo} (Dataset Treino)', fontsize=16, y=1.03)

        for i in range(num_exemplos_plotar):
            ax = plt.subplot(num_exemplos_plotar, 1, i + 1)
            dados_segmento = segmentos_exemplo[i].numpy()
            label_codificado = np.argmax(labels_exemplo_one_hot[i].numpy())
            nome_gesto = nomes_classes_ordenados[label_codificado]

            plot_data_subset = dados_segmento[:, :num_canais_plotar]
            offset_vertical = 1.0
            if plot_data_subset.size > 0:
                range_plot = np.ptp(plot_data_subset)
                max_abs = np.max(np.abs(plot_data_subset))
                if range_plot > 1e-9:
                    offset_vertical = range_plot * 0.8
                elif max_abs > 1e-9:
                    offset_vertical = max_abs

            for ch_idx_plot in range(num_canais_plotar):
                offset = ch_idx_plot * offset_vertical
                ax.plot(eixo_tempo, dados_segmento[:, ch_idx_plot] + offset,
                                 label=f'EMG {INDICES_CANAIS_EMG[ch_idx_plot]+1}', alpha=0.85, lw=1.2)

            label_eixo_y = 'Amplitude (+Offset)' if USAR_DADOS_BRUTOS else 'Amplitude Norm. (+Offset)'
            ax.set_title(f'Exemplo {i+1} - Gesto: "{nome_gesto}" (Label Codificado: {label_codificado})', fontsize=12)
            ax.set_xlabel('Tempo (ms)', fontsize=10)
            ax.set_ylabel(label_eixo_y, fontsize=10)
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small', title="Canais EMG")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(labelsize=9)
        plt.tight_layout(rect=[0, 0, 0.93, 0.98])
        plt.show()
    except Exception as e:
        print(f"\nAVISO: Falha ao gerar plots de exemplo de segmentos: {e}")
        import traceback
        traceback.print_exc()
else:
    print("INFO: Plots de exemplo de segmentos não gerados pois o dataset de treino está vazio ou um modelo será carregado.")

print("\n--- Definição do Modelo CNN Base ---")
def criar_modelo_cnn_base(
    input_shape=(COMPRIMENTO_SEGMENTO, NUM_CANAIS_EMG), num_classes=NUM_CLASSES,
    filtros_conv1=64, kernel_size_conv1=9, activation='relu',
    dropout_rate=0.3, dense_units=256,
    optimizer_name='Adam', learning_rate=1e-3 ):

    entrada = layers.Input(shape=input_shape, name='Entrada_EMG')
    x = entrada
    # Bloco 1
    x = layers.Conv1D(filters=filtros_conv1, kernel_size=kernel_size_conv1, padding='same', name='Conv1_1')(x)
    x = layers.BatchNormalization(name='BN1_1')(x)
    x = layers.Activation(activation, name='Act1_1')(x)
    x = layers.Conv1D(filters=filtros_conv1, kernel_size=3, padding='same', name='Conv1_2')(x)
    x = layers.BatchNormalization(name='BN1_2')(x)
    x = layers.Activation(activation, name='Act1_2')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same', name='Pool1')(x)
    x = layers.Dropout(dropout_rate, name='Drop1')(x)

    filtros_conv2 = filtros_conv1 * 2
    x = layers.Conv1D(filters=filtros_conv2, kernel_size=5, padding='same', name='Conv2_1')(x)
    x = layers.BatchNormalization(name='BN2_1')(x)
    x = layers.Activation(activation, name='Act2_1')(x)
    x = layers.Conv1D(filters=filtros_conv2, kernel_size=3, padding='same', name='Conv2_2')(x)
    x = layers.BatchNormalization(name='BN2_2')(x)
    x = layers.Activation(activation, name='Act2_2')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same', name='Pool2')(x)
    x = layers.Dropout(dropout_rate, name='Drop2')(x)

    filtros_conv3 = filtros_conv2 * 2
    x = layers.Conv1D(filters=filtros_conv3, kernel_size=3, padding='same', name='Conv3_1')(x)
    x = layers.BatchNormalization(name='BN3_1')(x)
    x = layers.Activation(activation, name='Act3_1')(x)

    filtros_conv4 = filtros_conv3
    x = layers.Conv1D(filters=filtros_conv4, kernel_size=3, padding='same', name='Conv4_1')(x)
    x = layers.BatchNormalization(name='BN4_1')(x)
    x = layers.Activation(activation, name='Act4_1')(x)

    x = layers.GlobalAveragePooling1D(name='GlobalAvgPool')(x)
    x = layers.Dropout(min(0.9, dropout_rate + 0.1), name='Drop_PostConv')(x)

    x = layers.Dense(units=dense_units, name='Dense1')(x)
    x = layers.BatchNormalization(name='BN_Dense1')(x)
    x = layers.Activation(activation, name='Act_Dense1')(x)
    x = layers.Dropout(min(0.9, dropout_rate + 0.2), name='Drop_Dense1')(x)

    saida = layers.Dense(num_classes, activation='softmax', name='Saida_Softmax')(x)
    modelo = models.Model(inputs=entrada, outputs=saida, name="CNN_EMG_Otimizada")

    optimizer_class = DICIONARIO_OTIMIZADORES.get(optimizer_name, optimizers.Adam)
    opt = optimizer_class(learning_rate=learning_rate)

    modelo.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

melhor_modelo_final = None
history_of_best_model_refit = None
grid_search = None
melhores_hps_dict = {} # Inicializa para evitar NameError

if CAMINHO_MODELO_CARREGAR:
    print(f"\n--- Carregando Modelo Salvo de: '{CAMINHO_MODELO_CARREGAR}' ---")
    try:
        melhor_modelo_final = tf.keras.models.load_model(CAMINHO_MODELO_CARREGAR)
        print("INFO: Modelo carregado com sucesso.")
        print("\nArquitetura do Modelo Carregado:")
        melhor_modelo_final.summary(line_length=110)
        # Se você tiver os hiperparâmetros que foram usados para treinar este modelo,
        # você pode preenchê-los aqui para que sejam exibidos na seção de métricas.
        # Por exemplo:
        melhores_hps_dict = {
            'filtros_conv1': 64, # Exemplo, ajuste para o seu modelo salvo
            'activation': 'relu',
            'dropout_rate': 0.3,
            'optimizer_name': 'Adam',
            'learning_rate': 0.001,
            'dense_units': 256
        }
        print("INFO: Hiperparâmetros padrão assumidos para o modelo carregado. Ajuste 'melhores_hps_dict' se souber os valores exatos.")

    except Exception as e:
        print(f"ERRO ao carregar o modelo de '{CAMINHO_MODELO_CARREGAR}': {e}")
        import traceback
        traceback.print_exc()
        print("Prosseguindo com GridSearchCV e treinamento, pois o carregamento falhou.")
        CAMINHO_MODELO_CARREGAR = None # Reseta para forçar o treinamento
else:
    print("\nINFO: Nenhum caminho de modelo para carregar especificado. Iniciando GridSearchCV e Treinamento.")


if not CAMINHO_MODELO_CARREGAR:
    X_treino_np, y_treino_np = None, None
    X_val_np, y_val_np = None, None

    if dataset_treino:
        print("\nINFO: Convertendo dataset_treino (features e labels) para arrays NumPy para GridSearchCV...")
        X_treino_list = []
        y_treino_list = []
        for x_batch, y_batch in tqdm_notebook(dataset_treino, desc="Convertendo Treino para NumPy"):
            X_treino_list.append(x_batch.numpy())
            y_treino_list.append(y_batch.numpy())
        X_treino_np = np.concatenate(X_treino_list, axis=0)
        y_treino_np = np.concatenate(y_treino_list, axis=0)
        print(f"INFO: Shape X_treino_np: {X_treino_np.shape}, Shape y_treino_np: {y_treino_np.shape}")
    else:
        print("AVISO: Dataset de treino está vazio. Não será possível executar GridSearchCV.")

    if dataset_teste_eval:
        print("\nINFO: Convertendo dataset_teste_eval (features e labels) para arrays NumPy para validação GridSearchCV...")
        X_val_list = []
        y_val_list = []
        for x_batch, y_batch in tqdm_notebook(dataset_teste_eval, desc="Convertendo Validação para NumPy"):
            X_val_list.append(x_batch.numpy())
            y_val_list.append(y_batch.numpy())
        X_val_np = np.concatenate(X_val_list, axis=0)
        y_val_np = np.concatenate(y_val_list, axis=0)
        print(f"INFO: Shape X_val_np: {X_val_np.shape}, Shape y_val_np: {y_val_np.shape}")
    else:
        print("AVISO: Dataset de teste (para avaliação) está vazio. A validação no GridSearchCV não será possível.")

    if X_treino_np is None or X_val_np is None:
        print("ERRO CRÍTICO: Dados de treino ou de validação estão vazios (após conversão para NumPy). Não é possível prosseguir com GridSearchCV.")
        grid_search = None
        melhor_modelo_final = None
        history_of_best_model_refit = None
    else:
        print("\n--- Configuração do GridSearchCV ---")
        keras_clf = KerasClassifier(
            model=criar_modelo_cnn_base,
            input_shape=(COMPRIMENTO_SEGMENTO, NUM_CANAIS_EMG),
            num_classes=NUM_CLASSES,
            epochs=EPOCAS_GRID_SEARCH,
            batch_size=TAMANHO_LOTE,
            verbose=0,
        )
        param_grid = {
            'model__filtros_conv1': FILTROS_CONV1_GRID,
            'model__activation': FUNCOES_ATIVACAO_GRID,
            'model__dropout_rate': DROPOUT_RATES_GRID,
            'model__optimizer_name': OPCOES_OTIMIZADORES_GRID,
            'model__learning_rate': TAXAS_APRENDIZADO_GRID,
            'model__dense_units': [128, 256]
        }
        print("\nINFO: Grid de Hiperparâmetros para GridSearchCV:")
        for key, value in param_grid.items(): print(f"  - {key}: {value}")

        es_callback_grid = EarlyStopping( monitor='val_loss', patience=PACIENCIA_EARLY_STOPPING // 2, verbose=0, mode='min', restore_best_weights=True )
        rlr_callback_grid = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=PACIENCIA_REDUCE_LR // 2, min_lr=1e-7, verbose=0, mode='min' )
        callbacks_grid = [es_callback_grid, rlr_callback_grid]

        grid_search = GridSearchCV(
            estimator=keras_clf, param_grid=param_grid, scoring='accuracy',
            cv=3,
            refit=True, verbose=2, n_jobs=1, error_score='raise'
        )
        num_combinations = 1
        for p_values in param_grid.values(): num_combinations *= len(p_values)

        print(f"\nINFO: Iniciando GridSearchCV... (Testando {num_combinations} combinações)")
        print(f"      Cada combinação será treinada por até {EPOCAS_GRID_SEARCH} épocas.")
        print(f"      Usando X_treino_np para treinar e (X_val_np, y_val_np) para validação.")
        print("ATENÇÃO: Esta etapa pode ser demorada!")

        t_grid_inicio = time.time()
        try:
            grid_search.fit( X=X_treino_np, y=y_treino_np, validation_data=(X_val_np, y_val_np), callbacks=callbacks_grid )
            t_grid_fim = time.time()
            print(f"\nGridSearchCV CONCLUÍDO em {(t_grid_fim - t_grid_inicio) / 60:.2f} minutos.")
            print("\n--- Resultados do GridSearchCV e Modelo Final ---")
            print(f"\nMelhor Acurácia de Validação (segundo GridSearchCV): {grid_search.best_score_:.4f}")
            print("\nMelhores Hiperparâmetros encontrados:")
            melhores_hps_dict_raw = grid_search.best_params_
            melhores_hps_dict = {k.split('__')[-1]: v for k, v in melhores_hps_dict_raw.items()}
            for param, value in melhores_hps_dict.items(): print(f"  * {param:<25}: {value}")

            melhor_modelo_final = grid_search.best_estimator_.model_
            history_of_best_model_refit = grid_search.best_estimator_.history_
            print("\nArquitetura do Melhor Modelo Encontrado:")
            if melhor_modelo_final: melhor_modelo_final.summary(line_length=110)

        except Exception as e_grid:
            print(f"ERRO durante GridSearchCV: {e_grid}")
            import traceback
            traceback.print_exc()
            grid_search = None
            melhor_modelo_final = None
            history_of_best_model_refit = None

if grid_search and history_of_best_model_refit and hasattr(history_of_best_model_refit, 'history'):
    print("\n--- Curvas de Treinamento do Melhor Modelo (Refit) ---")
    history_dict = history_of_best_model_refit.history
    plt.figure(figsize=(14, 5))
    plot_items = [('loss', 'Perda Treino', 'val_loss', 'Perda Validação', 'Histórico de Perda'),
                  ('accuracy', 'Acurácia Treino', 'val_accuracy', 'Acurácia Validação', 'Histórico de Acurácia')]
    for i, (train_metric, train_label, val_metric, val_label, title) in enumerate(plot_items):
        plt.subplot(1, 2, i + 1)
        if train_metric in history_dict: plt.plot(history_dict[train_metric], label=train_label)
        if val_metric in history_dict: plt.plot(history_dict[val_metric], label=val_label)
        plt.title(title + ' (Melhor Modelo Refit)')
        plt.xlabel('Época'); plt.ylabel(train_metric.capitalize())
        plt.legend(); plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(); plt.show()
else:
    print("AVISO: Histórico de treinamento do melhor modelo (refit) não disponível para plotagem.")


if melhor_modelo_final and dataset_teste_metrics:
    print("\n--- Avaliação Final do Melhor Modelo ---")
    print("INFO: Coletando predições no conjunto de teste (usando dataset_teste_metrics)...")
    t_aval_i = time.time()
    labels_reais_lista, predicoes_prob_lista = [], []

    for segmentos_lote, labels_reais_lote_encoded in tqdm_notebook(dataset_teste_metrics, desc="Avaliando Lotes Teste"):
        prob_lote = melhor_modelo_final.predict_on_batch(segmentos_lote)
        predicoes_prob_lista.append(prob_lote.numpy())
        labels_reais_lista.append(labels_reais_lote_encoded.numpy())

    y_prob_preditas_final = np.concatenate(predicoes_prob_lista, axis=0)
    y_labels_reais_final_encoded = np.concatenate(labels_reais_lista, axis=0)
    y_labels_preditos_final_encoded = np.argmax(y_prob_preditas_final, axis=1)

    t_aval_f = time.time()
    print(f"INFO: Avaliação no conjunto de teste concluída em {t_aval_f - t_aval_i:.2f} seg.")
    print(f"      Total de {len(y_labels_reais_final_encoded)} amostras de teste avaliadas.")

    y_labels_reais_final_oh = to_categorical(y_labels_reais_final_encoded, num_classes=NUM_CLASSES)

    print("\n" + "="*30 + " MÉTRICAS DE DESEMPENHO FINAL " + "="*30)
    if melhores_hps_dict:
        print("INFO: Métricas referentes ao modelo com os seguintes hiperparâmetros:")
        for param, value in melhores_hps_dict.items(): print(f"  - {param:<25}: {value}")
    print("-" * 80)

    acc = accuracy_score(y_labels_reais_final_encoded, y_labels_preditos_final_encoded)
    prec_w = precision_score(y_labels_reais_final_encoded, y_labels_preditos_final_encoded, average='weighted', zero_division=0)
    rec_w = recall_score(y_labels_reais_final_encoded, y_labels_preditos_final_encoded, average='weighted', zero_division=0)
    f1_w = f1_score(y_labels_reais_final_encoded, y_labels_preditos_final_encoded, average='weighted', zero_division=0)

    auc_roc_ovr_w, logloss_val, shannon, mse, rmse, mae, r2 = (float('nan'),)*7
    sens_media, espec_media = float('nan'), float('nan')
    cm_final = None

    try: auc_roc_ovr_w = roc_auc_score(y_labels_reais_final_oh, y_prob_preditas_final, multi_class='ovr', average='weighted')
    except ValueError as e: print(f"AVISO AUC ROC: {e}")
    try: logloss_val = log_loss(y_labels_reais_final_oh, y_prob_preditas_final)
    except ValueError as e: print(f"AVISO Log Loss: {e}")
    try: shannon = calcular_entropia_shannon(y_prob_preditas_final)
    except Exception as e: print(f"AVISO Entropia Shannon: {e}")

    mse = mean_squared_error(y_labels_reais_final_encoded, y_labels_preditos_final_encoded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_labels_reais_final_encoded, y_labels_preditos_final_encoded)
    try: r2 = r2_score(y_labels_reais_final_encoded, y_labels_preditos_final_encoded)
    except ValueError as e: print(f"AVISO R² score: {e}")

    try:
        cm_final = confusion_matrix(y_labels_reais_final_encoded, y_labels_preditos_final_encoded, labels=range(NUM_CLASSES))
        sens_p_classe, espec_p_classe = calcular_sens_espec(cm_final)
        sens_media = np.mean(sens_p_classe[~np.isnan(sens_p_classe)])
        espec_media = np.mean(espec_p_classe[~np.isnan(espec_p_classe)])
    except Exception as e_ss: print(f"AVISO Sens/Espec: {e_ss}")

    print(f"\nAcurácia Geral: {acc:.4f}")
    print("\nRelatório de Classificação Detalhado (por Gesto):")
    print(classification_report(y_labels_reais_final_encoded, y_labels_preditos_final_encoded,
                                 target_names=nomes_classes_ordenados,
                                 labels=range(NUM_CLASSES), zero_division=0))
    print("\nMétricas Adicionais Calculadas:")
    metricas_finais_dict = {
        'Acurácia': acc, 'Precisão (Pond)': prec_w, 'Recall (Pond)': rec_w, 'F1-Score (Pond)': f1_w,
        'AUC ROC (OvR Pond)': auc_roc_ovr_w, 'Log Loss': logloss_val,
        'Sensibilidade Média': sens_media, 'Especificidade Média': espec_media,
        'Entropia Shannon Média': shannon, 'MSE (Classes)': mse,
        'RMSE (Classes)': rmse, 'MAE (Classes)': mae, 'R² (Classes)': r2
    }
    for nome, valor in metricas_finais_dict.items(): print(f"  - {nome:<30}: {'N/A' if np.isnan(valor) else f'{valor:.4f}'}")

    print("\n" + "="*30 + " VISUALIZAÇÕES FINAIS " + "="*30)
    if cm_final is not None:
        print("\nGerando Matriz de Confusão...")
        try:
            fig_cm, ax_cm = plt.subplots(figsize=(max(8, NUM_CLASSES * 1.2), max(7, NUM_CLASSES * 1.0)))
            disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=nomes_classes_ordenados)
            disp_cm.plot(cmap='viridis', ax=ax_cm, xticks_rotation=45, colorbar=True, values_format='d')
            ax_cm.set_title('Matriz de Confusão Final', fontsize=16)
            ax_cm.set_xlabel('Gesto Predito', fontsize=13); ax_cm.set_ylabel('Gesto Real', fontsize=13)
            plt.tight_layout(); plt.show()
        except Exception as e: print(f"ERRO Matriz de Confusão: {e}\n{traceback.format_exc()}")
    else: print("\nAVISO: Matriz de Confusão não pôde ser gerada.")

    print("\nGerando Curvas ROC (One-vs-Rest)...")
    try:
        plt.figure(figsize=(max(10, NUM_CLASSES*0.9), max(9, NUM_CLASSES*0.8)))
        try: cores = plt.cm.get_cmap('tab10', NUM_CLASSES)
        except AttributeError: cores = plt.colormaps.get_cmap('tab10').resampled(NUM_CLASSES)
        fpr, tpr, roc_auc_val_dict = {}, {}, {}
        for i in range(NUM_CLASSES):
            nome_classe = nomes_classes_ordenados[i]
            if np.sum(y_labels_reais_final_oh[:, i]) > 0 and np.sum(y_labels_reais_final_oh[:, i] != 1) > 0 :
                fpr[i], tpr[i], _ = roc_curve(y_labels_reais_final_oh[:, i], y_prob_preditas_final[:, i])
                roc_auc_val_dict[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], color=cores(i), lw=2.5, alpha=0.85,
                                 label=f'{nome_classe} (AUC = {roc_auc_val_dict[i]:.3f})')
            else: print(f"AVISO ROC: Classe '{nome_classe}' (Cod: {i}) não tem amostras suficientes de ambas as classes no teste. Curva não gerada.")
        plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle=':', label='Aleatório (AUC = 0.500)')
        plt.xlim([-0.05, 1.0]); plt.ylim([-0.05, 1.05])
        plt.xlabel('Taxa de Falsos Positivos', fontsize=14); plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
        plt.title('Curvas ROC (One-vs-Rest)', fontsize=17)
        plt.legend(loc="lower right", fontsize='medium', title="Classes e AUC", title_fontsize='13')
        plt.grid(True, linestyle='--', alpha=0.6); plt.tick_params(labelsize=11); plt.show()
    except Exception as e: print(f"ERRO Curvas ROC: {e}\n{traceback.format_exc()}")
else:
    print("\nINFO: Avaliação Final não realizada (modelo ou dados de teste não disponíveis).")

if melhor_modelo_final and not CAMINHO_MODELO_CARREGAR: # Só salva se um novo modelo foi treinado
    print("\n--- Salvar o Melhor Modelo ---")
    try:
        melhor_modelo_final.save(NOME_ARQUIVO_MODELO_SALVO)
        print(f"\nMelhor modelo salvo com sucesso como: '{NOME_ARQUIVO_MODELO_SALVO}'")
        try: print(f"  Localização: {os.path.abspath(NOME_ARQUIVO_MODELO_SALVO)}")
        except: pass
    except Exception as e: print(f"\nERRO ao salvar o modelo final: {e}")
elif melhor_modelo_final and CAMINHO_MODELO_CARREGAR:
    print(f"\nINFO: O modelo carregado de '{CAMINHO_MODELO_CARREGAR}' não será salvo novamente.")
else:
    print("\nAVISO: Nenhum modelo final foi treinado ou encontrado para salvar.")
