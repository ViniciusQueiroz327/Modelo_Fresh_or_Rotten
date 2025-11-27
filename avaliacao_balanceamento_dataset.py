import os
import shutil
import kagglehub
import uuid


print("📥 Baixando dataset via kagglehub...")

dataset_path = kagglehub.dataset_download(
    "narayanibokde/augmented-dataset-for-fruits-rottenfresh"
)

print("Path:", dataset_path)


# =====================================================================
# 1. LOCALIZAR AUTOMATICAMENTE AS PASTAS FreshX e RottenX
# =====================================================================

def localizar_pastas_brutas(path):
    """
    Localiza as pastas originais do dataset (FreshApple, RottenApple etc)
    """
    candidatos = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d.lower().startswith("fresh") or d.lower().startswith("rotten"):
                candidatos.append(os.path.join(root, d))
    return candidatos


originais = localizar_pastas_brutas(dataset_path)

if len(originais) == 0:
    raise RuntimeError("❌ Não foram encontradas pastas FreshX/RottenX no dataset!")

print("\n📂 Pastas detectadas:")
for p in originais:
    print(" -", p)


# =====================================================================
# 2. CRIAR A NOVA ESTRUTURA fresh/ e rotten/
# =====================================================================

BASE_ORGANIZADA = os.path.join(dataset_path, "ORGANIZADO")
fresh_dir = os.path.join(BASE_ORGANIZADA, "fresh")
rotten_dir = os.path.join(BASE_ORGANIZADA, "rotten")

os.makedirs(fresh_dir, exist_ok=True)
os.makedirs(rotten_dir, exist_ok=True)

print("\n📦 Criando pasta organizada em:", BASE_ORGANIZADA)


# =====================================================================
# 3. MOVER TODAS IMAGENS PARA fresh/ ou rotten/
# =====================================================================

def mover_imagens(pasta):
    nome = os.path.basename(pasta).lower()

    if nome.startswith("fresh"):
        destino = fresh_dir
    elif nome.startswith("rotten"):
        destino = rotten_dir
    else:
        return

    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for f in arquivos:
        origem = os.path.join(pasta, f)

        # gerar nome único para evitar colisões entre frutas
        novo_nome = f"{uuid.uuid4().hex}.jpg"

        destino_final = os.path.join(destino, novo_nome)
        shutil.copy2(origem, destino_final)


for pasta in originais:
    mover_imagens(pasta)

print("\n✔ Todas as imagens foram reorganizadas com sucesso!")
print(f"➡ fresh/: {len(os.listdir(fresh_dir))} imagens")
print(f"➡ rotten/: {len(os.listdir(rotten_dir))} imagens")


# =====================================================================
# 4. ANÁLISE FINAL (contagem + amostras)
# =====================================================================

import cv2
import matplotlib.pyplot as plt
import random

def contar(path):
    return len([f for f in os.listdir(path) if f.lower().endswith((".jpg",".jpeg",".png"))])

fresh_count = contar(fresh_dir)
rotten_count = contar(rotten_dir)

print("\n=== Contagem final ===")
print(f"Fresh: {fresh_count}")
print(f"Rotten: {rotten_count}")
print(f"Total: {fresh_count + rotten_count}")


# Exibir amostras
plt.figure(figsize=(10, 5))

for i, classe_dir in enumerate([fresh_dir, rotten_dir]):
    imgs = [os.path.join(classe_dir, f) for f in os.listdir(classe_dir)]
    samples = random.sample(imgs, 3)

    for j, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 3, i * 3 + j + 1)
        plt.imshow(img)
        plt.title("Fresh" if i == 0 else "Rotten")
        plt.axis("off")

plt.tight_layout()
plt.show()

print("\n🎉 Reorganização e análise concluídas!")