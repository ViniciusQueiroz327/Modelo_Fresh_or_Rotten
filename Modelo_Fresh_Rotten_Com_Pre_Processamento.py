import os
import shutil
import uuid
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import cv2
import kagglehub
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import time
import seaborn as sns
import hashlib

# ===============================================================================
# RODAR NO GOOGLE COLAB
# https://colab.research.google.com/drive/1qGWvekc4GR4pXq_vz3xFTGpCMO8AU3Rj?usp=sharing
# ===============================================================================

# ===============================================================================
# 0. PACOTES OBRIGATÓRIOS
#   pip install numpy
#   pip install matplotlib
#   pip install scikit-learn
#   pip install opencv-python
#   pip install seaborn
#   pip install kagglehub
#   
#   pip install numpy matplotlib scikit-learn opencv-python seaborn kagglehub
# 
#   Python 3.13.2:
#       python --version
# ===============================================================================

# ============================================================== 
# 1. BAIXAR DATASET
# ==============================================================

print("📥 Baixando dataset via kagglehub...")

dataset_path = kagglehub.dataset_download(
    "narayanibokde/augmented-dataset-for-fruits-rottenfresh"
)

print("📦 Dataset baixado em:", dataset_path)


# -----------------------------
# Limpeza automática de versões antigas do kagglehub
# -----------------------------
def _find_kagglehub_version_parent(path):
    cur = os.path.dirname(path)
    # sobe até raiz, procurando pasta que contenha entradas 'v*'
    while True:
        try:
            entries = os.listdir(cur)
        except Exception:
            return None
        versions = [e for e in entries if e.startswith("v")]
        if len(versions) > 1:
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent

def limpar_kagglehub_versions(path_dataset):
    base_dir = _find_kagglehub_version_parent(path_dataset)
    if base_dir is None:
        print("Nenhuma pasta de versões do KaggleHub encontrada para limpeza.")
        return

    versões = sorted(
        [v for v in os.listdir(base_dir) if v.startswith("v")],
        reverse=True
    )

    if len(versões) <= 1:
        print("Nenhuma versão extra do KaggleHub para limpar.")
        return

    print("\n🧹 Limpando versões antigas do KaggleHub...\n")

    # Mantém só a primeira (mais recente)
    versões_para_apagar = versões[1:]

    for v in versões_para_apagar:
        caminho = os.path.join(base_dir, v)
        try:
            shutil.rmtree(caminho, ignore_errors=True)
            print(f"   🔥 Removido: {caminho}")
        except Exception as e:
            print(f"   ⚠ Falha ao remover {caminho}: {e}")

    print("\n✔ Versões antigas removidas com sucesso!\n")


# chama limpeza automática
try:
    limpar_kagglehub_versions(dataset_path)
except Exception as e:
    print("⚠ Erro ao tentar limpar versões do KaggleHub:", e)


# ============================================================== 
# 2. ENCONTRAR PASTAS ORIGINAIS
# ==============================================================

def localizar_pastas_brutas(path):
    candidatos = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            nome = d.lower()
            if nome.startswith("fresh") or nome.startswith("rotten"):
                candidatos.append(os.path.join(root, d))
    return candidatos

originais = localizar_pastas_brutas(dataset_path)

print("\n📂 Pastas detectadas:")
for p in originais:
    print(" -", p)


# ============================================================== 
# 3. ORGANIZAR EM fresh/ rotten/ (com limpeza antes)
# ==============================================================

BASE_ORGANIZADA = os.path.join(dataset_path, "ORGANIZADO")
fresh_dir  = os.path.join(BASE_ORGANIZADA, "fresh")
rotten_dir = os.path.join(BASE_ORGANIZADA, "rotten")

# 🔥 LIMPEZA PARA EVITAR DUPLICAÇÃO
shutil.rmtree(BASE_ORGANIZADA, ignore_errors=True)
os.makedirs(fresh_dir, exist_ok=True)
os.makedirs(rotten_dir, exist_ok=True)

print("\n🧹 Limpando e reorganizando imagens...")

# -----------------------------
# Funções para copiar sem duplicar (baseadas em hash)
# -----------------------------
def file_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def build_hash_set(folder):
    hashes = set()
    if not os.path.exists(folder):
        return hashes
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith((".jpg",".jpeg",".png")):
            try:
                hashes.add(file_md5(p))
            except Exception:
                # ignora arquivos ilegíveis
                pass
    return hashes

# inicializa conjuntos de hashes para evitar duplicação
hashes_fresh = build_hash_set(fresh_dir)
hashes_rotten = build_hash_set(rotten_dir)

def copiar_sem_duplicar_por_hash(origem, destino_dir, hashes_set):
    try:
        md5 = file_md5(origem)
    except Exception:
        return False
    if md5 in hashes_set:
        return False
    novo_nome = f"{uuid.uuid4().hex}.jpg"
    destino_final = os.path.join(destino_dir, novo_nome)
    shutil.copy2(origem, destino_final)
    hashes_set.add(md5)
    return True

# agora mover_imagens usa cópia sem duplicação por hash
total_copiadas_fresh = 0
total_copiadas_rotten = 0
copiados_por_origem = {}  # para log: {origem_dir: n_copiadas}

def mover_imagens(pasta):
    global total_copiadas_fresh, total_copiadas_rotten
    nome = os.path.basename(pasta).lower()
    if nome.startswith("fresh"):
        destino = fresh_dir
        hashes_set = hashes_fresh
    else:
        destino = rotten_dir
        hashes_set = hashes_rotten

    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    copiadas = 0

    for f in arquivos:
        origem = os.path.join(pasta, f)
        if copiar_sem_duplicar_por_hash(origem, destino, hashes_set):
            copiadas += 1

    copiados_por_origem[pasta] = copiadas
    if nome.startswith("fresh"):
        total_copiadas_fresh += copiadas
    else:
        total_copiadas_rotten += copiadas

for pasta in originais:
    mover_imagens(pasta)

print("\n✔ Reorganização concluída!")
print(f"➡ fresh/ (novas cópias):  {total_copiadas_fresh} imagens")
print(f"➡ rotten/ (novas cópias): {total_copiadas_rotten} imagens")

# Exibe detalhamento por pasta de origem
print("\n📋 Detalhe de cópias por pasta de origem:")
for origem_pasta, qtd in copiados_por_origem.items():
    print(f" - {os.path.basename(origem_pasta)}: {qtd} imagens copiadas")


# ============================================================== 
# 4. CRIAR train/ val/ test (com limpeza antes)
# ==============================================================

base_out = "/content/dataset"

train_path = os.path.join(base_out, "train")
val_path   = os.path.join(base_out, "val")
test_path  = os.path.join(base_out, "test")

# 🔥 LIMPA TUDO ANTES DE CRIAR
shutil.rmtree(base_out, ignore_errors=True)

for folder in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(folder, "fresh"), exist_ok=True)
    os.makedirs(os.path.join(folder, "rotten"), exist_ok=True)


def split_dataset(src, train_dir, val_dir, test_dir, split=(0.7, 0.15, 0.15)):
    files = [f for f in os.listdir(src) if f.lower().endswith(("jpg","png","jpeg"))]
    random.shuffle(files)

    total = len(files)
    n_train = int(split[0] * total)
    n_val   = int(split[1] * total)

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    for f in train_files:
        shutil.copy2(os.path.join(src, f), os.path.join(train_dir, f))

    for f in val_files:
        shutil.copy2(os.path.join(src, f), os.path.join(val_dir, f))

    for f in test_files:
        shutil.copy2(os.path.join(src, f), os.path.join(test_dir, f))


split_dataset(fresh_dir,  os.path.join(train_path,"fresh"), 
                           os.path.join(val_path,"fresh"),
                           os.path.join(test_path,"fresh"))

split_dataset(rotten_dir, os.path.join(train_path,"rotten"), 
                           os.path.join(val_path,"rotten"),
                           os.path.join(test_path,"rotten"))


# ============================================================== 
# 5. CARREGAMENTO + PRÉ-PROCESSAMENTO
# ==============================================================

IMG_SIZE = (128, 128)

# Pré-Processamento
def load_one_image(path_label):
    path, label = path_label
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE) # redimensionamento
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale --> Alteração para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converte para RGB
    img = img.astype("float32") / 255.0 # normaliza
    return img.flatten(), label # flatten agora contém 3 canais

def carregar_pasta_threads(pasta, label):
    files = [
        (os.path.join(pasta, f), label)
        for f in os.listdir(pasta)
        if f.lower().endswith(("jpg","jpeg","png"))
    ]

    X, y = [], []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_one_image, fl) for fl in files]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                img, lbl = result
                X.append(img)
                y.append(lbl)

    return np.array(X), np.array(y)


def carregar_dataset(base):
    xf, yf = carregar_pasta_threads(os.path.join(base, "fresh"), 0)
    xr, yr = carregar_pasta_threads(os.path.join(base, "rotten"), 1)
    X = np.concatenate([xf, xr])
    y = np.concatenate([yf, yr])
    return X, y


print("\n⚡ Carregando imagens (multithreaded)...")

X_train, y_train = carregar_dataset(train_path)
X_val,   y_val   = carregar_dataset(val_path)
X_test,  y_test  = carregar_dataset(test_path)


# ============================================================== 
# 5.1 VISUALIZAÇÃO APÓS PRÉ-PROCESSAMENTO
# ==============================================================

print("\n🔍 Exibindo amostras pós pré-processamento...")

def show_examples(X, y):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Amostras pós pré-processamento (128x128x3 RGB)")

    classes = [0, 1]  # fresh=0, rotten=1
    titles = ["Fresh", "Rotten"]

    for cls in classes:
        idx = np.where(y == cls)[0][:5]
        for i, img_idx in enumerate(idx):
            # img = X[img_idx].reshape(128, 128) # Para grayscale --> Mudamos para RGB
            img = X[img_idx].reshape(128, 128, 3) # Para RGB
            ax = axes[cls][i]
            ax.imshow(img, cmap="gray")
            ax.set_title(titles[cls])
            ax.axis("off")

    plt.show()

show_examples(X_train, y_train)

#========================================================================================================================================================================
# Alteração de GRAYSCALE para RGB pois em termos de podridão em frutas, as manchas na casca podem variar muito de cor e tom, e usar GRAYSCALE fazia com que o modelo
# muitas vezes identificasse manchas mais leves como sombra / podridão, aumentando o número de FN (Fresh, mas era Rotten) e FP (Rotten, mas era Fresh)
# Utilizar o RGB diminui um pouco a quantidade de FN e FP
#========================================================================================================================================================================


# ============================================================== 
# 6. NORMALIZAÇÃO + SVM
# ==============================================================

pipeline = Pipeline([
    ("pca", PCA(n_components=150, whiten=True, random_state=42)),
    ("svm", LinearSVC(C=1.0, max_iter=5000))
])

print("\n⏳ Treinando PCA + SVM...")
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()

print(f"\n✅ Treinamento concluído em {end_time - start_time:.2f} segundos")

acc = pipeline.score(X_test, y_test)
print(f"🎯 Acurácia: {acc:.4f}")


# ============================================================== 
# 7. MÉTRICAS
# ==============================================================

# Precision → quando o modelo diz fresh e realmente é fresh
# Recall → quanto ele encontra de todas as imagens realmente fresh
# F1-score → equilíbrio entre precisão e recall

print("\n📈 Avaliando modelo...")

# LinearSVC não tem predict_proba → usar decision_function()
scores = pipeline.decision_function(X_test)
y_pred = (scores > 0).astype(int) # --> Bom equilíbrio
# y_pred = (scores > -0.2).astype(int) # --> Aumentando sensibilidade do modelo para detectar ROTTEN --> Descartado pois aumentou consideravelmente a taxa de desperdício
# y_pred = (scores > -0.5).astype(int) # --> Aumentando sensibilidade do modelo para detectar ROTTEN --> Taxa de desperdício e segurança ALTAS
# y_pred = (scores > -1.0).astype(int) # --> Aumentando sensibilidade do modelo para detectar ROTTEN --> Vai priorizar SEGURANÇA antes do DESPERDÍCIO Taxa de desperdício e segurança MUITO ALTAS

#========================================================================================================================================================================
# Cada aumento no treshold acima faz com que o modelo classifique mais Fresh como Rotten e menos Rotten como Fresh.
# Portanto quando menor o valor de corte, maior vai ser o desperdício e menor vai ser o risco, por classificar mais Fresh como Rotten.
# E quanto maior for o valor de corte, menor vai ser o desperdício e maior vai ser o risco, por classificar menos Fresh como Rotten.

# y_pred = (scores > 0).astype(int) # --> Visando o objetivo do projeto, esse treshold será priorizado, pois balanceia desperdício e segurança.
#========================================================================================================================================================================

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, target_names=["Fresh", "Rotten"]))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# Rótulos do eixo
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusão com VP / FP / FN / VN")

# Substitui os rótulos numéricos pelos nomes das classes
ax.set_xticklabels(["Fresh", "Rotten"])
ax.set_yticklabels(["Fresh", "Rotten"])

# --- Adiciona texto explicando VP / FP / FN / VN em cada célula ---
# cm = [[TN, FP],
#       [FN, TP]]

TN, FP = cm[0]
FN, TP = cm[1]

ax.text(0.5, 0.5, "VN\n(Verdadeiro Negativo)", ha="center", va="center", fontsize=10, color="black")
ax.text(1.5, 0.5, "FP\n(Falso Positivo)", ha="center", va="center", fontsize=10, color="black")
ax.text(0.5, 1.5, "FN\n(Falso Negativo)", ha="center", va="center", fontsize=10, color="black")
ax.text(1.5, 1.5, "VP\n(Verdadeiro Positivo)", ha="center", va="center", fontsize=10, color="black")

plt.show()

# VP –-> Verdadeiro Positivo –-> Predito Rotten e era Rotten
# FP –-> Falso Positivo	Predito –-> Rotten, mas era Fresh
# TN –-> Verdadeiro Negativo –-> Predito Fresh e era Fresh
# FN –-> Falso Negativo	Predito –-> Fresh, mas era Rotten

fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.title("Curva ROC")
plt.xlabel("Falso Positivo")
plt.ylabel("Verdadeiro Positivo")
plt.legend()
plt.show()

#=========================================================================================
# ROC:
#   Ela mostra como o modelo se comporta variando o limiar ("threshold") de decisão.
#       Se score > 0 → predito Rotten (classe positiva)
#       Se score <= 0 → predito Fresh
#   A curva ROC junta todos os possíveis thresholds e plota:
#       Eixo X: FPR (Falso Positivo)
#       Eixo Y: TPR (Verdadeiro Positivo)

# "Rotten" é a classe positiva:
# TPR alto → modelo detecta frutas podres corretamente
# FPR baixo → modelo quase não marca fruitas boas como podres
# 90%+ de sensibilidade (boa detecção do podre)
# 9–10% de FP (marcar algumas frutas frescas como podres)

# Essa alta taxa AUC tem possíveis causas:
#   Balanceamento muito bom entre as classes Rotten e Fresh
#   Dataset é bem limpo --> com fundo uniforme em grande parte das imagens, 
#   muitas frutas isoladas, pouca sujeira visual, entre outros
#   Resolução consistente das imagens do dataset
#=========================================================================================