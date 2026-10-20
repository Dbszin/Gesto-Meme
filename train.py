"""
train.py - Treina um classificador de gestos com Transfer Learning (MobileNetV2).

Estrutura esperada:
    dataset/
        joinha/       *.jpg
        continencia/  *.jpg
        neutro/       *.jpg   (sem gesto, recomendado)
        ...

Saídas:
    modelo_gestos.h5  -> modelo treinado
    classes.json      -> mapeamento índice -> {nome, emoji}

Uso:
    python train.py
"""

import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
PASTA_DATASET = Path("dataset")
ARQUIVO_MODELO = "modelo_gestos.h5"
ARQUIVO_CLASSES = "classes.json"

TAMANHO_IMG = 224          # MobileNetV2 foi pré-treinada em 224x224
BATCH = 32
VALIDACAO = 0.2            # 20% das imagens vão para validação
SEED = 42
EPOCAS_TOPO = 15           # fase 1: treina só a "cabeça" nova
EPOCAS_FINE_TUNING = 10    # fase 2: ajusta também as últimas camadas da base
CAMADAS_DESCONGELADAS = 30
MIN_IMAGENS_POR_CLASSE = 50

# Emoji RESERVA de cada gesto (nome da pasta -> emoji).
# O app.py mostra primeiro a imagem referencias/<nome>.jpeg; este emoji só
# aparece se o gesto não tiver imagem lá.
# Pastas que não estiverem aqui recebem "❓". Use "" para não exibir nada.
EMOJIS = {
    "joinha": "👍",
    "oculos": "🤓",
    "desconfiado": "😒",
    "pensativo": "🤔",
    "continencia": "🫡",
    "caramba": "😜",
    "neutro": "",
}


def verificar_dataset():
    """Valida a estrutura de pastas e avisa sobre classes com poucas imagens."""
    if not PASTA_DATASET.is_dir():
        raise SystemExit(f"Pasta '{PASTA_DATASET}' não encontrada. Rode o coleta.py primeiro.")

    extensoes = {".jpg", ".jpeg", ".png", ".bmp"}
    contagem = {
        p.name: sum(1 for f in p.iterdir() if f.suffix.lower() in extensoes)
        for p in sorted(PASTA_DATASET.iterdir()) if p.is_dir()
    }
    if len(contagem) < 2:
        raise SystemExit("São necessárias pelo menos 2 pastas (classes) dentro de dataset/.")

    print("Imagens por classe:")
    for nome, qtd in contagem.items():
        aviso = "  <- poucas imagens!" if qtd < MIN_IMAGENS_POR_CLASSE else ""
        print(f"  {nome:15s} {qtd:5d}{aviso}")


def carregar_datasets():
    """Lê as pastas e divide em treino/validação."""
    treino, validacao = keras.utils.image_dataset_from_directory(
        PASTA_DATASET,
        validation_split=VALIDACAO,
        subset="both",
        seed=SEED,
        image_size=(TAMANHO_IMG, TAMANHO_IMG),
        batch_size=BATCH,
        label_mode="int",
    )
    nomes_classes = treino.class_names  # ordem alfabética das pastas

    # cache + prefetch deixam o treino bem mais rápido
    autotune = tf.data.AUTOTUNE
    treino = treino.cache().shuffle(1000).prefetch(autotune)
    validacao = validacao.cache().prefetch(autotune)
    return treino, validacao, nomes_classes


def construir_modelo(num_classes):
    """MobileNetV2 congelada + data augmentation + cabeça de classificação."""
    # Data Augmentation: só atua durante o treino (desligado na predição)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),          # ±36 graus
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    base = keras.applications.MobileNetV2(
        input_shape=(TAMANHO_IMG, TAMANHO_IMG, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    entradas = keras.Input(shape=(TAMANHO_IMG, TAMANHO_IMG, 3))
    x = augmentation(entradas)
    # Normaliza pixels de [0, 255] para [-1, 1], como a MobileNetV2 espera
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    # training=False mantém o BatchNorm da base em modo inferência,
    # o que é importante para o fine-tuning não "estragar" os pesos
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    saidas = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(entradas, saidas), base


def treinar(modelo, treino, validacao, epocas, taxa_aprendizado):
    modelo.compile(
        optimizer=keras.optimizers.Adam(taxa_aprendizado),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, verbose=1),
    ]
    modelo.fit(treino, validation_data=validacao, epochs=epocas, callbacks=callbacks)


def salvar_classes(nomes_classes):
    mapa = {
        "tamanho_imagem": TAMANHO_IMG,
        "classes": {
            str(i): {"nome": nome, "emoji": EMOJIS.get(nome, "❓")}
            for i, nome in enumerate(nomes_classes)
        },
    }
    with open(ARQUIVO_CLASSES, "w", encoding="utf-8") as f:
        json.dump(mapa, f, ensure_ascii=False, indent=2)


def main():
    verificar_dataset()
    treino, validacao, nomes_classes = carregar_datasets()
    print(f"Classes: {nomes_classes}")

    modelo, base = construir_modelo(len(nomes_classes))
    modelo.summary()

    # Fase 1: treina apenas a camada Dense nova (base congelada)
    print("\n=== Fase 1: treinando a cabeça do modelo ===")
    treinar(modelo, treino, validacao, EPOCAS_TOPO, 1e-3)

    # Fase 2: fine-tuning das últimas camadas da MobileNetV2 com LR baixa
    print("\n=== Fase 2: fine-tuning ===")
    base.trainable = True
    for camada in base.layers[:-CAMADAS_DESCONGELADAS]:
        camada.trainable = False
    treinar(modelo, treino, validacao, EPOCAS_FINE_TUNING, 1e-5)

    perda, acuracia = modelo.evaluate(validacao, verbose=0)
    print(f"\nValidação final -> perda: {perda:.4f} | acurácia: {acuracia:.2%}")

    modelo.save(ARQUIVO_MODELO)
    salvar_classes(nomes_classes)
    print(f"Modelo salvo em '{ARQUIVO_MODELO}' e classes em '{ARQUIVO_CLASSES}'.")


if __name__ == "__main__":
    main()
