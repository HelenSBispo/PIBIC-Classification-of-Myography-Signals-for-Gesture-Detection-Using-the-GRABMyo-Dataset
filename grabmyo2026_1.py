# =============================================================================
# CNN EMG - GrabMyo (16 gestos + Repouso = 17 classes)
# Otimizado para KAGGLE com sistema de RETOMADA DE TREINAMENTO ROBUSTO

# =============================================================================
# INSTRUÇÕES PARA KAGGLE:
#   1. Adicione o dataset GrabMyo ao notebook (Add Data -> Search "grabmyo")
#      ou faça upload e crie um Dataset privado.
#   2. Ative a GPU: Settings -> Accelerator -> GPU T4 x2 (ou P100)
#   3. Ative Persistence: Settings -> Persistence -> Files only
#   4. Execute. Se a sessão cair, basta executar novamente: o treinamento
#      retoma da última época salva automaticamente.
# =============================================================================

!pip install wfdb -q

#===============================================================================
import os
import gc
import json
import time
import math
import random
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import wfdb
import scipy.signal
from scipy.stats import mode as scipy_mode, entropy as scipy_entropy
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score, log_loss,
    cohen_kappa_score, matthews_corrcoef
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, models, regularizers, callbacks
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# =============================================================================
# 1. REPRODUTIBILIDADE & GPU
# =============================================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print(" CONFIGURAÇÃO DE GPU E AMBIENTE ".center(70, "="))
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ TensorFlow detectou {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        # Mixed precision: ~2x mais rápido em GPUs modernas (T4, P100, V100, A100)
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✓ Mixed Precision (float16) ATIVADO para acelerar treinamento.")
        except Exception as e:
            print(f"  Aviso: Mixed precision não habilitado ({e})")
    except RuntimeError as e:
        print(f"⚠ Erro ao configurar GPU: {e}")
else:
    print("⚠ Nenhuma GPU detectada. O treinamento será MUITO mais lento na CPU.")
    print("  No Kaggle: Settings -> Accelerator -> GPU T4 x2")

print(f"✓ TensorFlow versão: {tf.__version__}")
print(f"✓ Seed fixada: {SEED}")

# =============================================================================
# 2. CAMINHOS (KAGGLE / COLAB / LOCAL)
# =============================================================================
def detectar_ambiente_e_caminhos():
    """Detecta automaticamente Kaggle, Colab ou local e retorna caminhos."""
    # Kaggle
    if os.path.exists('/kaggle/input'):
        print("✓ Ambiente detectado: KAGGLE")
        # Procura automaticamente a pasta do GrabMyo dentro de /kaggle/input
        candidatos = []
        for root_dir in os.listdir('/kaggle/input'):
            full_path = os.path.join('/kaggle/input', root_dir)
            for sub_root, dirs, files in os.walk(full_path):
                if any(d.startswith('Session') for d in dirs):
                    candidatos.append(sub_root)
                    break
        if not candidatos:
            raise FileNotFoundError(
                "Não encontrei pasta com 'SessionX' em /kaggle/input/. "
                "Adicione o dataset GrabMyo ao notebook."
            )
        caminho_dados = candidatos[0]
        caminho_saida = '/kaggle/working'
        return caminho_dados, caminho_saida

    # Google Colab
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive', force_remount=True)
        print("✓ Ambiente detectado: COLAB")
        caminho_dados = '/content/drive/MyDrive/GrabMyo/physionet.org/files/grabmyo/1.1.0'
        caminho_saida = '/content/drive/MyDrive/GrabMyo/checkpoints'
        os.makedirs(caminho_saida, exist_ok=True)
        return caminho_dados, caminho_saida
    except ModuleNotFoundError:
        pass

    # Local
    print("✓ Ambiente detectado: LOCAL")
    caminho_dados = 'D:/Dowloads/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.0'
    caminho_saida = './checkpoints'
    os.makedirs(caminho_saida, exist_ok=True)
    return caminho_dados, caminho_saida

CAMINHO_BASE_DADOS, CAMINHO_SAIDA = detectar_ambiente_e_caminhos()

if not os.path.exists(CAMINHO_BASE_DADOS):
    raise FileNotFoundError(f"⚠ Caminho não encontrado: {CAMINHO_BASE_DADOS}")
print(f"✓ Dados:  {CAMINHO_BASE_DADOS}")
print(f"✓ Saída:  {CAMINHO_SAIDA}")

# =============================================================================
# 3. CONSTANTES E HIPERPARÂMETROS
# =============================================================================
# --- Sinal EMG ---
FREQ_AMOSTRAGEM     = 2048           # Hz
DURACAO_SEGMENTO_MS = 250            # ms (segmentos curtos = mais amostras)
COMPRIMENTO_SEGMENTO = int(FREQ_AMOSTRAGEM * (DURACAO_SEGMENTO_MS / 1000))
SOBREPOSICAO        = 0.75           # 75% overlap → ~4x mais segmentos
PASSO_JANELA        = max(1, int(COMPRIMENTO_SEGMENTO * (1 - SOBREPOSICAO)))

# --- Filtragem ---
APLICAR_FILTRO      = True
FREQ_CORTE_INFERIOR = 20.0
FREQ_CORTE_SUPERIOR = 500.0
FREQ_NOTCH          = 60.0           # Remove rede elétrica
ORDEM_FILTRO        = 4

# --- Canais EMG ---
# GrabMyo tem 28 canais EMG (16 antebraço + 12 punho) + canais auxiliares.
# O canal de label costuma ser o último. Ajuste conforme seu arquivo.
CANAIS_TOTAIS_ESPERADOS = 32
INDICES_CANAIS_EMG  = list(range(28))   # Usar TODOS os canais EMG p/ máxima informação
INDICE_CANAL_LABEL  = 31
NUM_CANAIS_EMG      = len(INDICES_CANAIS_EMG)

# --- Classes (17 = 16 gestos + repouso) ---
DICIONARIO_CLASSES_GESTOS = {
    0:  'Repouso',
    1:  'Lateral Grasp',
    2:  'Thumb Adduction',
    3:  'Thumb-Index Pinch',
    4:  'Thumb-Middle Pinch',
    5:  'Thumb-Ring Pinch',
    6:  'Thumb-Little Pinch',
    7:  'Index Extension',
    8:  'Middle Extension',
    9:  'Ring Extension',
    10: 'Little Extension',
    11: 'Wrist Flexion',
    12: 'Wrist Extension',
    13: 'Radial Deviation',
    14: 'Ulnar Deviation',
    15: 'Hand Grasp',
    16: 'Hand Open',
}

# --- Treinamento ---
TAMANHO_LOTE         = 128
EPOCAS_TOTAIS        = 80
PACIENCIA_ES         = 15
PACIENCIA_RLR        = 6
LR_INICIAL           = 1e-3
LR_MIN               = 1e-7
PROPORCAO_VAL        = 0.15
PROPORCAO_TESTE      = 0.15
LABEL_SMOOTHING      = 0.05
WEIGHT_DECAY         = 1e-4
USAR_AUGMENTATION    = True
USAR_CACHE_NPZ       = True   # Cache em disco evita reprocessar dados

# --- Arquitetura ---
FILTROS_BASE         = 64
DROPOUT_RATE         = 0.4
DENSE_UNITS          = 256

# --- Caminhos de arquivos ---
PATH_CACHE_DADOS     = os.path.join(CAMINHO_SAIDA, 'cache_dados_grabmyo.npz')
PATH_CHECKPOINT_BEST = os.path.join(CAMINHO_SAIDA, 'modelo_emg_best.keras')
PATH_CHECKPOINT_LAST = os.path.join(CAMINHO_SAIDA, 'modelo_emg_last.keras')
PATH_HISTORICO_CSV   = os.path.join(CAMINHO_SAIDA, 'historico_treinamento.csv')
PATH_ESTADO_TREINO   = os.path.join(CAMINHO_SAIDA, 'estado_treinamento.json')
PATH_LABEL_ENCODER   = os.path.join(CAMINHO_SAIDA, 'label_encoder.npy')

# Detecta sessões disponíveis automaticamente
SESSOES_A_CARREGAR = sorted([
    d for d in os.listdir(CAMINHO_BASE_DADOS)
    if d.lower().startswith('session') and os.path.isdir(os.path.join(CAMINHO_BASE_DADOS, d))
])
if not SESSOES_A_CARREGAR:
    raise ValueError(f"Nenhuma pasta 'SessionX' encontrada em {CAMINHO_BASE_DADOS}")

print("\n" + "=" * 70)
print(" CONFIGURAÇÃO ".center(70, "="))
print("=" * 70)
print(f"  Sessões:                {SESSOES_A_CARREGAR}")
print(f"  Freq. amostragem:       {FREQ_AMOSTRAGEM} Hz")
print(f"  Segmento:               {COMPRIMENTO_SEGMENTO} amostras ({DURACAO_SEGMENTO_MS} ms)")
print(f"  Overlap:                {SOBREPOSICAO*100:.0f}% (passo: {PASSO_JANELA})")
print(f"  Canais EMG:             {NUM_CANAIS_EMG}")
print(f"  Filtragem:              Bandpass {FREQ_CORTE_INFERIOR}-{FREQ_CORTE_SUPERIOR}Hz + Notch {FREQ_NOTCH}Hz")
print(f"  Augmentation:           {USAR_AUGMENTATION}")
print(f"  Batch size:             {TAMANHO_LOTE}")
print(f"  Épocas (max):           {EPOCAS_TOTAIS}")
print(f"  Learning rate:          {LR_INICIAL} → cosine decay")
print(f"  Mixed precision:        {tf.keras.mixed_precision.global_policy().name}")

# =============================================================================
# 4. PRÉ-PROCESSAMENTO DE SINAL
# =============================================================================
def filtrar_emg(dados, fs=FREQ_AMOSTRAGEM):
    """Bandpass Butterworth + Notch para 60 Hz (rede elétrica)."""
    nyq = 0.5 * fs
    low  = max(FREQ_CORTE_INFERIOR / nyq, 0.001)
    high = min(FREQ_CORTE_SUPERIOR / nyq, 0.999)
    try:
        # Bandpass
        sos_bp = scipy.signal.butter(ORDEM_FILTRO, [low, high], btype='band', output='sos')
        dados_f = scipy.signal.sosfiltfilt(sos_bp, dados, axis=0)
        # Notch 60 Hz
        b_notch, a_notch = scipy.signal.iirnotch(FREQ_NOTCH, Q=30, fs=fs)
        dados_f = scipy.signal.filtfilt(b_notch, a_notch, dados_f, axis=0)
        return dados_f
    except Exception as e:
        print(f"  Aviso filtro: {e}")
        return dados

def normalizar_zscore(sinal):
    """Z-score por canal (axis=0) com proteção contra std~0."""
    media = np.mean(sinal, axis=0, keepdims=True)
    dp = np.std(sinal, axis=0, keepdims=True)
    dp[dp < 1e-8] = 1.0
    return (sinal - media) / dp

# =============================================================================
# 5. EXTRAÇÃO DE SEGMENTOS (com janelamento overlap + cache em disco)
# =============================================================================
def extrair_segmentos_de_arquivo(caminho_sem_ext):
    """Extrai todos segmentos válidos de um arquivo .dat/.hea com overlap."""
    segmentos, labels = [], []
    try:
        registro = wfdb.rdrecord(caminho_sem_ext, warn_empty=True)
        if registro.fs != FREQ_AMOSTRAGEM:                  return [], []
        if registro.p_signal is None:                       return [], []
        if registro.sig_len < COMPRIMENTO_SEGMENTO:         return [], []
        if registro.n_sig != CANAIS_TOTAIS_ESPERADOS:       return [], []

        sinais = registro.p_signal
        if np.isnan(sinais).any() or np.isinf(sinais).any():
            sinais = np.nan_to_num(sinais, nan=0.0, posinf=0.0, neginf=0.0)

        sinais_emg = sinais[:, INDICES_CANAIS_EMG].astype(np.float32)
        labels_tempo = sinais[:, INDICE_CANAL_LABEL].astype(np.int32)

        # Filtragem + normalização (uma vez por arquivo)
        if APLICAR_FILTRO:
            sinais_emg = filtrar_emg(sinais_emg, fs=registro.fs)
        sinais_emg = normalizar_zscore(sinais_emg)

        # Janelamento com overlap
        n = sinais_emg.shape[0]
        for inicio in range(0, n - COMPRIMENTO_SEGMENTO + 1, PASSO_JANELA):
            fim = inicio + COMPRIMENTO_SEGMENTO
            seg = sinais_emg[inicio:fim, :]
            lbls_seg = labels_tempo[inicio:fim]

            # Label do segmento = moda. Aceita só labels válidos.
            lbls_validos = lbls_seg[
                np.isin(lbls_seg, list(DICIONARIO_CLASSES_GESTOS.keys()))
            ]
            if lbls_validos.size == 0:
                continue
            mode_res = scipy_mode(lbls_validos, keepdims=False)
            lbl = int(mode_res.mode if np.isscalar(mode_res.mode) else mode_res.mode.item())

            # Garante que o label predomina no segmento (>50%) - mais limpo
            if np.sum(lbls_seg == lbl) / len(lbls_seg) < 0.5:
                continue

            segmentos.append(seg.astype(np.float32))
            labels.append(np.int32(lbl))
    except Exception as e:
        # Silencioso: arquivos quebrados são ignorados
        pass
    return segmentos, labels

def coletar_todos_dados():
    """
    Percorre TODAS as sessões e arquivos, extrai segmentos.
    Salva em .npz para acelerar próximas execuções.
    """
    if USAR_CACHE_NPZ and os.path.exists(PATH_CACHE_DADOS):
        print(f"\n✓ Carregando cache: {PATH_CACHE_DADOS}")
        try:
            dados = np.load(PATH_CACHE_DADOS, allow_pickle=False)
            X, y = dados['X'], dados['y']
            print(f"   Shape X: {X.shape}, Shape y: {y.shape}")
            print(f"   Distribuição de classes: {dict(zip(*np.unique(y, return_counts=True)))}")
            return X, y
        except Exception as e:
            print(f"  Cache corrompido ({e}). Reprocessando...")

    print("\n" + "=" * 70)
    print(" EXTRAINDO SEGMENTOS DOS ARQUIVOS .dat ".center(70, "="))
    print("=" * 70)
    print("  (Esta etapa só roda 1x. Resultado é salvo em cache.)")

    arquivos_para_processar = []
    for sessao in SESSOES_A_CARREGAR:
        sess_path = os.path.join(CAMINHO_BASE_DADOS, sessao)
        for root, _, files in os.walk(sess_path):
            for f in files:
                if f.endswith('.dat'):
                    arquivos_para_processar.append(
                        os.path.join(root, f.replace('.dat', ''))
                    )
    arquivos_para_processar.sort()
    print(f"  Arquivos encontrados: {len(arquivos_para_processar)}")

    todos_X, todos_y = [], []
    for caminho in tqdm(arquivos_para_processar, desc="Processando arquivos"):
        segs, lbls = extrair_segmentos_de_arquivo(caminho)
        if segs:
            todos_X.extend(segs)
            todos_y.extend(lbls)

    X = np.array(todos_X, dtype=np.float32)
    y = np.array(todos_y, dtype=np.int32)
    del todos_X, todos_y
    gc.collect()

    print(f"\n✓ Total de segmentos: {len(y)}")
    print(f"   Shape X: {X.shape}, Memória: {X.nbytes / 1e9:.2f} GB")
    print(f"   Distribuição: {dict(zip(*np.unique(y, return_counts=True)))}")

    if USAR_CACHE_NPZ:
        try:
            print(f"  Salvando cache em {PATH_CACHE_DADOS}...")
            np.savez(PATH_CACHE_DADOS, X=X, y=y)
            print("  ✓ Cache salvo.")
        except Exception as e:
            print(f"  Aviso: não foi possível salvar cache ({e})")
    return X, y

X_full, y_full = coletar_todos_dados()
if len(X_full) == 0:
    raise ValueError("Nenhum segmento foi extraído! Verifique caminhos e estrutura dos dados.")

# =============================================================================
# 6. CODIFICAÇÃO DE LABELS E SPLIT TREINO/VAL/TESTE
# =============================================================================
print("\n" + "=" * 70)
print(" CODIFICAÇÃO E SPLIT DOS DADOS ".center(70, "="))
print("=" * 70)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_full)
classes_originais = label_encoder.classes_
NUM_CLASSES = len(classes_originais)
nomes_classes_ordenados = [
    DICIONARIO_CLASSES_GESTOS.get(int(c), f"Classe_{c}")
    for c in classes_originais
]
np.save(PATH_LABEL_ENCODER, classes_originais)

print(f"  Classes detectadas ({NUM_CLASSES}):")
for i, nome in enumerate(nomes_classes_ordenados):
    n = int(np.sum(y_encoded == i))
    print(f"    [{i:2d}] {nome:<25} → {n:>6d} segmentos")

# Split estratificado: treino/val/teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X_full, y_encoded, test_size=PROPORCAO_TESTE,
    stratify=y_encoded, random_state=SEED
)
val_relativo = PROPORCAO_VAL / (1 - PROPORCAO_TESTE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_relativo,
    stratify=y_temp, random_state=SEED
)
del X_temp, y_temp, X_full, y_full, y_encoded
gc.collect()

print(f"\n  Split:  Treino={len(X_train)}  |  Val={len(X_val)}  |  Teste={len(X_test)}")

# Pesos das classes (caso desbalanceadas)
class_weights_arr = compute_class_weight(
    'balanced', classes=np.arange(NUM_CLASSES), y=y_train
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
print(f"  Pesos das classes (balanced): {[f'{w:.2f}' for w in class_weights_arr]}")

# =============================================================================
# 7. DATA AUGMENTATION (em GPU via tf.data)
# =============================================================================
@tf.function
def augment_emg(sinal, label):
    """Augmentation EMG: ruído + scaling + shifting + channel dropout."""
    # 1) Gaussian noise (SNR ~20 dB)
    if tf.random.uniform([]) < 0.5:
        ruido = tf.random.normal(tf.shape(sinal), mean=0.0, stddev=0.05)
        sinal = sinal + ruido
    # 2) Magnitude scaling
    if tf.random.uniform([]) < 0.5:
        escala = tf.random.uniform([1, tf.shape(sinal)[1]], 0.85, 1.15)
        sinal = sinal * escala
    # 3) Time shifting
    if tf.random.uniform([]) < 0.3:
        shift = tf.random.uniform([], -10, 10, dtype=tf.int32)
        sinal = tf.roll(sinal, shift=shift, axis=0)
    # 4) Channel dropout (zera canal aleatório)
    if tf.random.uniform([]) < 0.2:
        n_canais = tf.shape(sinal)[1]
        idx = tf.random.uniform([], 0, n_canais, dtype=tf.int32)
        mask = tf.one_hot(idx, n_canais, on_value=0.0, off_value=1.0)
        sinal = sinal * mask
    return sinal, label

def fazer_dataset(X, y, treino=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if treino:
        ds = ds.shuffle(buffer_size=min(10000, len(X)), seed=SEED, reshuffle_each_iteration=True)
        if USAR_AUGMENTATION:
            ds = ds.map(augment_emg, num_parallel_calls=tf.data.AUTOTUNE)
    # Converte label inteiro -> one-hot
    ds = ds.map(
        lambda x, lab: (x, tf.one_hot(lab, depth=NUM_CLASSES, dtype=tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(TAMANHO_LOTE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

print("\n  Construindo tf.data pipelines...")
ds_train = fazer_dataset(X_train, y_train, treino=True)
ds_val   = fazer_dataset(X_val,   y_val,   treino=False)
ds_test  = fazer_dataset(X_test,  y_test,  treino=False)
print("  ✓ Pipelines prontos.")

# =============================================================================
# 8. ARQUITETURA: SE-ResNet 1D para EMG
# =============================================================================
def squeeze_excitation_block(x, reduction=16, name=''):
    """Squeeze-and-Excitation: re-pondera canais por importância."""
    canais = x.shape[-1]
    se = layers.GlobalAveragePooling1D(name=f'{name}_se_gap')(x)
    se = layers.Dense(max(canais // reduction, 4), activation='relu', name=f'{name}_se_d1')(se)
    se = layers.Dense(canais, activation='sigmoid', name=f'{name}_se_d2')(se)
    se = layers.Reshape((1, canais), name=f'{name}_se_rs')(se)
    return layers.Multiply(name=f'{name}_se_mul')([x, se])

def residual_block(x, filters, kernel_size=5, stride=1, name=''):
    """Bloco residual 1D com SE."""
    shortcut = x
    # Conv 1
    y = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                      name=f'{name}_c1')(x)
    y = layers.BatchNormalization(name=f'{name}_bn1')(y)
    y = layers.Activation('elu', name=f'{name}_a1')(y)
    # Conv 2
    y = layers.Conv1D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                      name=f'{name}_c2')(y)
    y = layers.BatchNormalization(name=f'{name}_bn2')(y)
    # SE
    y = squeeze_excitation_block(y, name=name)
    # Shortcut com adaptação de dimensão se necessário
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 name=f'{name}_sc')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_sc_bn')(shortcut)
    y = layers.Add(name=f'{name}_add')([y, shortcut])
    y = layers.Activation('elu', name=f'{name}_a2')(y)
    return y

def construir_modelo(input_shape, num_classes):
    """SE-ResNet 1D otimizada para classificação EMG multi-canal."""
    entrada = layers.Input(shape=input_shape, name='entrada_emg')

    # Stem
    x = layers.Conv1D(FILTROS_BASE, 11, strides=1, padding='same', name='stem_conv')(entrada)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('elu', name='stem_act')(x)
    x = layers.MaxPooling1D(2, padding='same', name='stem_pool')(x)

    # Stages
    x = residual_block(x, FILTROS_BASE,    name='r1a')
    x = residual_block(x, FILTROS_BASE,    name='r1b')
    x = layers.MaxPooling1D(2, padding='same', name='p1')(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5, name='d1')(x)

    x = residual_block(x, FILTROS_BASE * 2, stride=1, name='r2a')
    x = residual_block(x, FILTROS_BASE * 2,            name='r2b')
    x = layers.MaxPooling1D(2, padding='same', name='p2')(x)
    x = layers.Dropout(DROPOUT_RATE * 0.6, name='d2')(x)

    x = residual_block(x, FILTROS_BASE * 4, stride=1, name='r3a')
    x = residual_block(x, FILTROS_BASE * 4,            name='r3b')
    x = layers.MaxPooling1D(2, padding='same', name='p3')(x)
    x = layers.Dropout(DROPOUT_RATE * 0.7, name='d3')(x)

    x = residual_block(x, FILTROS_BASE * 8, stride=1, name='r4a')

    # Head
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dropout(DROPOUT_RATE, name='d_head')(x)
    x = layers.Dense(DENSE_UNITS, kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                     name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Activation('elu', name='act_dense1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='d_dense1')(x)

    # Saída em float32 (importante com mixed precision)
    saida = layers.Dense(num_classes, activation='softmax', dtype='float32', name='saida')(x)

    return models.Model(entrada, saida, name='SE_ResNet1D_EMG')

# =============================================================================
# 9. CALLBACKS - Sistema de RETOMADA ROBUSTA
# =============================================================================
class SalvarEstadoTreinamento(callbacks.Callback):
    """
    Salva estado completo do treinamento a cada época, permitindo retomar
    exatamente de onde parou caso a sessão Kaggle seja interrompida.
    """
    def __init__(self, path_estado, path_historico, path_last_model):
        super().__init__()
        self.path_estado    = path_estado
        self.path_historico = path_historico
        self.path_last      = path_last_model
        self.historico_acumulado = []
        # Carrega histórico anterior se existir
        if os.path.exists(path_historico):
            try:
                df = pd.read_csv(path_historico)
                self.historico_acumulado = df.to_dict('records')
                print(f"  ✓ Histórico anterior carregado: {len(self.historico_acumulado)} épocas")
            except Exception:
                pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 1) Salva modelo "last"
        try:
            self.model.save(self.path_last)
        except Exception as e:
            print(f"  Aviso: falha ao salvar last model ({e})")

        # 2) Acumula e salva histórico em CSV
        registro = {'epoca': epoch + 1, **{k: float(v) for k, v in logs.items()}}
        self.historico_acumulado.append(registro)
        try:
            pd.DataFrame(self.historico_acumulado).to_csv(self.path_historico, index=False)
        except Exception:
            pass

        # 3) Salva metadados de retomada
        try:
            estado = {
                'ultima_epoca': epoch + 1,
                'epocas_totais': EPOCAS_TOTAIS,
                'best_val_loss': min(r.get('val_loss', float('inf'))
                                     for r in self.historico_acumulado),
                'best_val_acc':  max(r.get('val_accuracy', 0)
                                     for r in self.historico_acumulado),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(self.path_estado, 'w') as f:
                json.dump(estado, f, indent=2)
        except Exception:
            pass

class CosineAnnealingComWarmup(callbacks.Callback):
    """LR scheduler com warmup + cosine annealing."""
    def __init__(self, lr_max, lr_min, total_epocas, warmup_epocas=3, epoca_inicial=0):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total = total_epocas
        self.warmup = warmup_epocas
        self.epoca_inicial = epoca_inicial

    def on_epoch_begin(self, epoch, logs=None):
        e = epoch + self.epoca_inicial
        if e < self.warmup:
            lr = self.lr_max * (e + 1) / self.warmup
        else:
            progresso = (e - self.warmup) / max(1, self.total - self.warmup)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progresso))
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except Exception:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if epoch % 5 == 0:
            print(f"  [LR Scheduler] Época {e+1}: lr = {lr:.2e}")

# =============================================================================
# 10. TREINAMENTO (com retomada automática)
# =============================================================================
print("\n" + "=" * 70)
print(" TREINAMENTO ".center(70, "="))
print("=" * 70)

# Tenta retomar de checkpoint
modelo = None
epoca_inicial = 0
if os.path.exists(PATH_CHECKPOINT_LAST) and os.path.exists(PATH_ESTADO_TREINO):
    try:
        with open(PATH_ESTADO_TREINO, 'r') as f:
            estado = json.load(f)
        epoca_inicial = estado.get('ultima_epoca', 0)
        if epoca_inicial < EPOCAS_TOTAIS:
            print(f"  ✓ Retomando treinamento da época {epoca_inicial + 1}/{EPOCAS_TOTAIS}")
            print(f"     Best val_acc anterior: {estado.get('best_val_acc', 0):.4f}")
            modelo = tf.keras.models.load_model(PATH_CHECKPOINT_LAST)
            print("  ✓ Modelo carregado de checkpoint.")
        else:
            print(f"  ✓ Treinamento já estava completo ({epoca_inicial} épocas).")
            modelo = tf.keras.models.load_model(PATH_CHECKPOINT_BEST)
            epoca_inicial = EPOCAS_TOTAIS  # pula treinamento
    except Exception as e:
        print(f"  Aviso: não foi possível retomar ({e}). Iniciando do zero.")
        modelo = None
        epoca_inicial = 0

if modelo is None:
    print("  Construindo modelo do zero...")
    modelo = construir_modelo(
        input_shape=(COMPRIMENTO_SEGMENTO, NUM_CANAIS_EMG),
        num_classes=NUM_CLASSES
    )
    optimizer = optimizers.AdamW(learning_rate=LR_INICIAL, weight_decay=WEIGHT_DECAY)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    modelo.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    print(f"  ✓ Modelo compilado. Parâmetros: {modelo.count_params():,}")

modelo.summary(line_length=110)

# Callbacks
cb_best = callbacks.ModelCheckpoint(
    PATH_CHECKPOINT_BEST, monitor='val_accuracy', mode='max',
    save_best_only=True, save_weights_only=False, verbose=1
)
cb_estado = SalvarEstadoTreinamento(PATH_ESTADO_TREINO, PATH_HISTORICO_CSV, PATH_CHECKPOINT_LAST)
cb_es = callbacks.EarlyStopping(
    monitor='val_accuracy', mode='max', patience=PACIENCIA_ES,
    restore_best_weights=True, verbose=1
)
cb_lr = CosineAnnealingComWarmup(LR_INICIAL, LR_MIN, EPOCAS_TOTAIS, warmup_epocas=3,
                                  epoca_inicial=epoca_inicial)
cb_lr_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=PACIENCIA_RLR,
    min_lr=LR_MIN, verbose=1
)
cb_terminate = callbacks.TerminateOnNaN()

todos_callbacks = [cb_best, cb_estado, cb_es, cb_lr, cb_terminate]

# Treina
if epoca_inicial < EPOCAS_TOTAIS:
    print(f"\n  Treinando da época {epoca_inicial + 1} até {EPOCAS_TOTAIS}...")
    t0 = time.time()
    history = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCAS_TOTAIS,
        initial_epoch=epoca_inicial,
        callbacks=todos_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    print(f"\n  ✓ Treinamento concluído em {(time.time() - t0)/60:.1f} minutos.")
else:
    history = None
    print("  ✓ Pulando treinamento (já completo).")

# Carrega o MELHOR modelo (não o último) para avaliação final
print("\n  Carregando MELHOR modelo para avaliação...")
if os.path.exists(PATH_CHECKPOINT_BEST):
    modelo = tf.keras.models.load_model(PATH_CHECKPOINT_BEST)
    print(f"  ✓ Modelo carregado de {PATH_CHECKPOINT_BEST}")

# =============================================================================
# 11. FUNÇÕES DE MÉTRICAS
# =============================================================================
def calcular_sens_espec(matriz_confusao):
    n = matriz_confusao.shape[0]
    sens, espec = np.zeros(n), np.zeros(n)
    for k in range(n):
        TP = matriz_confusao[k, k]
        FN = matriz_confusao[k, :].sum() - TP
        FP = matriz_confusao[:, k].sum() - TP
        TN = matriz_confusao.sum() - TP - FN - FP
        sens[k]  = TP / (TP + FN) if (TP + FN) > 0 else 0
        espec[k] = TN / (TN + FP) if (TN + FP) > 0 else 0
    return sens, espec

def entropia_shannon_media(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    ents = scipy_entropy(probs, axis=1, base=2)
    return float(np.mean(ents[np.isfinite(ents)]))

# =============================================================================
# 12. AVALIAÇÃO FINAL E PLOTS
# =============================================================================
print("\n" + "=" * 70)
print(" AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ".center(70, "="))
print("=" * 70)

# Predições
print("  Coletando predições...")
y_test_oh = to_categorical(y_test, num_classes=NUM_CLASSES)
y_prob = modelo.predict(ds_test, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# Garante mesmo tamanho (em caso de batches descartados)
n_min = min(len(y_test), len(y_pred))
y_test, y_pred, y_prob = y_test[:n_min], y_pred[:n_min], y_prob[:n_min]
y_test_oh = y_test_oh[:n_min]

# --- Métricas escalares ---
acc        = accuracy_score(y_test, y_pred)
prec_w     = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_w      = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_w       = f1_score(y_test, y_pred, average='weighted', zero_division=0)
prec_m     = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec_m      = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_m       = f1_score(y_test, y_pred, average='macro', zero_division=0)
kappa      = cohen_kappa_score(y_test, y_pred)
mcc        = matthews_corrcoef(y_test, y_pred)
top3_acc   = float(np.mean([y_test[i] in np.argsort(y_prob[i])[-3:] for i in range(len(y_test))]))

try:    auc_w = roc_auc_score(y_test_oh, y_prob, multi_class='ovr', average='weighted')
except: auc_w = float('nan')
try:    auc_m = roc_auc_score(y_test_oh, y_prob, multi_class='ovr', average='macro')
except: auc_m = float('nan')
try:    ll = log_loss(y_test_oh, y_prob)
except: ll = float('nan')
shannon = entropia_shannon_media(y_prob)

cm = confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES))
sens_per_class, espec_per_class = calcular_sens_espec(cm)

print("\n" + "─" * 70)
print(" RESULTADOS PRINCIPAIS ".center(70, "─"))
print("─" * 70)
metricas = {
    'Acurácia':                acc,
    'Top-3 Acurácia':          top3_acc,
    'Precisão (weighted)':     prec_w,
    'Recall (weighted)':       rec_w,
    'F1-Score (weighted)':     f1_w,
    'Precisão (macro)':        prec_m,
    'Recall (macro)':          rec_m,
    'F1-Score (macro)':        f1_m,
    'AUC ROC (weighted OvR)':  auc_w,
    'AUC ROC (macro OvR)':     auc_m,
    'Cohen Kappa':             kappa,
    'Matthews Corr. Coef.':    mcc,
    'Sensibilidade média':     float(np.mean(sens_per_class)),
    'Especificidade média':    float(np.mean(espec_per_class)),
    'Log Loss':                ll,
    'Entropia Shannon média':  shannon,
}
for nome, val in metricas.items():
    if isinstance(val, float) and not np.isnan(val):
        print(f"  {nome:<28}: {val:.4f}")
    else:
        print(f"  {nome:<28}: N/A")

# Salva métricas em JSON
try:
    with open(os.path.join(CAMINHO_SAIDA, 'metricas_finais.json'), 'w') as f:
        json.dump({k: (None if np.isnan(v) else v) for k, v in metricas.items()},
                  f, indent=2)
    print(f"\n  ✓ Métricas salvas em {os.path.join(CAMINHO_SAIDA, 'metricas_finais.json')}")
except Exception as e:
    print(f"  Aviso: {e}")

print("\n  Relatório de Classificação por Gesto:")
print(classification_report(y_test, y_pred,
      target_names=nomes_classes_ordenados, zero_division=0, digits=4))

# =============================================================================
# 13. VISUALIZAÇÕES
# =============================================================================
print("\n" + "=" * 70)
print(" GERANDO VISUALIZAÇÕES ".center(70, "="))
print("=" * 70)

# --- 13.1 Curvas de treinamento ---
def plot_historico(path_csv, path_out):
    if not os.path.exists(path_csv):
        return
    df = pd.read_csv(path_csv)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if 'loss' in df and 'val_loss' in df:
        axes[0].plot(df['epoca'], df['loss'], label='Treino', lw=2)
        axes[0].plot(df['epoca'], df['val_loss'], label='Validação', lw=2)
        axes[0].set_title('Histórico de Perda', fontsize=14)
        axes[0].set_xlabel('Época'); axes[0].set_ylabel('Loss')
        axes[0].legend(); axes[0].grid(alpha=0.4)

    if 'accuracy' in df and 'val_accuracy' in df:
        axes[1].plot(df['epoca'], df['accuracy'], label='Treino', lw=2)
        axes[1].plot(df['epoca'], df['val_accuracy'], label='Validação', lw=2)
        axes[1].axhline(y=0.96, color='r', ls='--', alpha=0.7, label='Meta 96%')
        axes[1].set_title('Histórico de Acurácia', fontsize=14)
        axes[1].set_xlabel('Época'); axes[1].set_ylabel('Accuracy')
        axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(path_out, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Salvo: {path_out}")

plot_historico(PATH_HISTORICO_CSV,
               os.path.join(CAMINHO_SAIDA, 'plot_historico.png'))

# --- 13.2 Matriz de confusão ---
fig, ax = plt.subplots(figsize=(max(10, NUM_CLASSES*0.7), max(8, NUM_CLASSES*0.6)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes_ordenados)
disp.plot(cmap='viridis', ax=ax, xticks_rotation=45, values_format='d', colorbar=True)
ax.set_title(f'Matriz de Confusão (Acurácia = {acc:.4f})', fontsize=15)
ax.set_xlabel('Gesto Predito', fontsize=12)
ax.set_ylabel('Gesto Real', fontsize=12)
plt.tight_layout()
path_cm = os.path.join(CAMINHO_SAIDA, 'plot_matriz_confusao.png')
plt.savefig(path_cm, dpi=130, bbox_inches='tight')
plt.show()
print(f"  ✓ Salvo: {path_cm}")

# Matriz normalizada
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
fig, ax = plt.subplots(figsize=(max(10, NUM_CLASSES*0.7), max(8, NUM_CLASSES*0.6)))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=nomes_classes_ordenados,
            yticklabels=nomes_classes_ordenados, ax=ax, cbar=True, vmin=0, vmax=1)
ax.set_title('Matriz de Confusão Normalizada (por linha)', fontsize=15)
ax.set_xlabel('Predito'); ax.set_ylabel('Real')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
path_cm_n = os.path.join(CAMINHO_SAIDA, 'plot_matriz_confusao_norm.png')
plt.savefig(path_cm_n, dpi=130, bbox_inches='tight')
plt.show()
print(f"  ✓ Salvo: {path_cm_n}")

# --- 13.3 Curvas ROC (One-vs-Rest) ---
fig, ax = plt.subplots(figsize=(11, 9))
try:    cores = plt.colormaps.get_cmap('tab20').resampled(NUM_CLASSES)
except: cores = plt.cm.get_cmap('tab20', NUM_CLASSES)

aucs_per_class = {}
for i in range(NUM_CLASSES):
    if y_test_oh[:, i].sum() == 0 or y_test_oh[:, i].sum() == len(y_test_oh):
        continue
    fpr_i, tpr_i, _ = roc_curve(y_test_oh[:, i], y_prob[:, i])
    auc_i = auc(fpr_i, tpr_i)
    aucs_per_class[nomes_classes_ordenados[i]] = auc_i
    ax.plot(fpr_i, tpr_i, color=cores(i), lw=2,
            label=f'{nomes_classes_ordenados[i]} (AUC={auc_i:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Aleatório (AUC=0.5)')
ax.set_xlim([-0.02, 1.0]); ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
ax.set_title('Curvas ROC One-vs-Rest (todas as classes)', fontsize=14)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(alpha=0.4)
plt.tight_layout()
path_roc = os.path.join(CAMINHO_SAIDA, 'plot_roc.png')
plt.savefig(path_roc, dpi=130, bbox_inches='tight')
plt.show()
print(f"  ✓ Salvo: {path_roc}")

# --- 13.4 Sensibilidade e Especificidade por classe ---
fig, ax = plt.subplots(figsize=(13, 6))
x_pos = np.arange(NUM_CLASSES)
w = 0.35
ax.bar(x_pos - w/2, sens_per_class,  w, label='Sensibilidade (Recall)', color='#2ecc71')
ax.bar(x_pos + w/2, espec_per_class, w, label='Especificidade',         color='#3498db')
ax.set_xticks(x_pos)
ax.set_xticklabels(nomes_classes_ordenados, rotation=45, ha='right')
ax.set_ylim(0, 1.05)
ax.axhline(y=0.96, color='r', ls='--', alpha=0.6, label='Meta 96%')
ax.set_ylabel('Valor')
ax.set_title('Sensibilidade e Especificidade por Gesto', fontsize=14)
ax.legend(); ax.grid(axis='y', alpha=0.4)
for i, (s, e) in enumerate(zip(sens_per_class, espec_per_class)):
    ax.text(i - w/2, s + 0.01, f'{s:.2f}', ha='center', fontsize=8)
    ax.text(i + w/2, e + 0.01, f'{e:.2f}', ha='center', fontsize=8)
plt.tight_layout()
path_sens = os.path.join(CAMINHO_SAIDA, 'plot_sens_espec.png')
plt.savefig(path_sens, dpi=130, bbox_inches='tight')
plt.show()
print(f"  ✓ Salvo: {path_sens}")

# --- 13.5 Distribuição de classes ---
fig, ax = plt.subplots(figsize=(13, 5))
counts = np.bincount(y_test, minlength=NUM_CLASSES)
ax.bar(nomes_classes_ordenados, counts, color='#e67e22')
ax.set_xticklabels(nomes_classes_ordenados, rotation=45, ha='right')
ax.set_ylabel('Nº de amostras (teste)')
ax.set_title('Distribuição de Classes no Conjunto de Teste', fontsize=14)
for i, c in enumerate(counts):
    ax.text(i, c + max(counts)*0.01, str(c), ha='center', fontsize=9)
plt.tight_layout()
path_dist = os.path.join(CAMINHO_SAIDA, 'plot_distribuicao.png')
plt.savefig(path_dist, dpi=130, bbox_inches='tight')
plt.show()
print(f"  ✓ Salvo: {path_dist}")

# =============================================================================
# 14. RESUMO FINAL
# =============================================================================
print("\n" + "=" * 70)
print(" RESUMO FINAL ".center(70, "="))
print("=" * 70)
print(f"  Acurácia final:          {acc*100:.2f}%")
print(f"  F1 (weighted):           {f1_w*100:.2f}%")
print(f"  AUC ROC (weighted):      {auc_w*100:.2f}%" if not np.isnan(auc_w) else "  AUC ROC: N/A")
print(f"  Sensibilidade média:     {np.mean(sens_per_class)*100:.2f}%")
print(f"  Especificidade média:    {np.mean(espec_per_class)*100:.2f}%")
print(f"  Cohen Kappa:             {kappa:.4f}")
print(f"  Matthews Corr. Coef.:    {mcc:.4f}")

print(f"\n  Arquivos salvos em: {CAMINHO_SAIDA}")
arquivos_gerados = [
    PATH_CHECKPOINT_BEST, PATH_CHECKPOINT_LAST, PATH_HISTORICO_CSV,
    PATH_ESTADO_TREINO, path_cm, path_cm_n, path_roc, path_sens, path_dist,
    os.path.join(CAMINHO_SAIDA, 'metricas_finais.json'),
    os.path.join(CAMINHO_SAIDA, 'plot_historico.png'),
]
for arq in arquivos_gerados:
    if os.path.exists(arq):
        print(f"    ✓ {os.path.basename(arq)}")

print("\n" + "=" * 70)
print(" PIPELINE FINALIZADO ".center(70, "="))
print("=" * 70)
