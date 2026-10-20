"""
app.py - Reconhece gestos pela webcam e mostra, ao lado do vídeo, a imagem
(meme) correspondente ao gesto.

A imagem de cada gesto é procurada em referencias/<nome do gesto>.jpeg/.jpg/.png
(ex: o gesto treinado em dataset/joinha/ mostra referencias/joinha.jpeg).
Se o gesto não tiver imagem, é mostrado o emoji definido no train.py.

Uso:
    python app.py
    python app.py --limiar 0.9 --camera 1

Tecla Q para sair.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

ARQUIVO_MODELO = "modelo_gestos.h5"
ARQUIVO_CLASSES = "classes.json"
PASTA_REFERENCIAS = Path("referencias")
LARGURA_PAINEL = 420   # largura do painel ao lado do vídeo, onde a imagem aparece
TAMANHO_EMOJI = 260
FRAMES_SUAVIZACAO = 5  # média das últimas N previsões, evita a imagem "piscar"

# Fontes de emoji coloridas (caminho, tamanho nativo). A Noto Color Emoji
# é bitmap (CBDT) e só aceita tamanho 109.
# ATENÇÃO: a Segoe UI Emoji do Windows 11 (seguiemj.ttf) usa o formato
# COLRv1, que o Pillow não renderiza (sai quadrado ou trava o Python),
# por isso usamos a Noto Color Emoji incluída na pasta fontes/.
FONTES_EMOJI = [
    (str(Path(__file__).parent / "fontes" / "NotoColorEmoji.ttf"), 109),
    ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109),  # Linux
    ("/System/Library/Fonts/Apple Color Emoji.ttc", 160),        # macOS
]


def carregar_fonte_emoji():
    for caminho, tamanho in FONTES_EMOJI:
        try:
            return ImageFont.truetype(caminho, tamanho)
        except OSError:
            continue
    raise SystemExit("Nenhuma fonte de emoji colorida encontrada no sistema.")


def renderizar_emoji(emoji, fonte, tamanho):
    """Desenha o emoji colorido com Pillow e devolve uma imagem BGRA (OpenCV)."""
    lado = int(fonte.size * 1.5)
    img = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(img)
    # embedded_color=True é o que faz o emoji sair colorido em vez de contorno
    desenho.text((lado // 2, lado // 2), emoji, font=fonte,
                 embedded_color=True, anchor="mm")
    caixa = img.getbbox()
    if caixa:
        img = img.crop(caixa)
    img = img.resize((tamanho, tamanho), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def carregar_referencia(nome):
    """Carrega a imagem (meme) do gesto, se existir em referencias/<nome>.*"""
    for ext in (".jpeg", ".jpg", ".png"):
        caminho = PASTA_REFERENCIAS / f"{nome}{ext}"
        if caminho.exists():
            img = cv2.imread(str(caminho))
            if img is not None:
                return img
    return None


def sobrepor(frame, imagem_bgra, x, y):
    """Cola uma imagem BGRA sobre o frame BGR respeitando a transparência."""
    h, w = imagem_bgra.shape[:2]
    h = min(h, frame.shape[0] - y)
    w = min(w, frame.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    recorte = imagem_bgra[:h, :w]
    alfa = recorte[:, :, 3:4].astype(np.float32) / 255.0
    regiao = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (alfa * recorte[:, :, :3] + (1 - alfa) * regiao).astype(np.uint8)


def montar_painel(altura, meme, emoji_bgra, texto, cor):
    """Cria o painel lateral com a imagem do gesto centralizada e o texto embaixo."""
    painel = np.full((altura, LARGURA_PAINEL, 3), 30, np.uint8)
    area_h = altura - 60  # espaço reservado para o texto

    if meme is not None:
        # Redimensiona mantendo a proporção para caber no painel
        h, w = meme.shape[:2]
        escala = min((LARGURA_PAINEL - 20) / w, (area_h - 20) / h)
        img = cv2.resize(meme, (int(w * escala), int(h * escala)))
        y = (area_h - img.shape[0]) // 2
        x = (LARGURA_PAINEL - img.shape[1]) // 2
        painel[y:y + img.shape[0], x:x + img.shape[1]] = img
    elif emoji_bgra is not None:
        lado = emoji_bgra.shape[0]
        sobrepor(painel, emoji_bgra, (LARGURA_PAINEL - lado) // 2, (area_h - lado) // 2)

    cv2.putText(painel, texto, (15, altura - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    return painel


def main():
    parser = argparse.ArgumentParser(description="Reconhecimento de gestos em tempo real")
    parser.add_argument("--limiar", type=float, default=0.80, help="Confiança mínima (0-1)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam")
    args = parser.parse_args()

    # Carrega modelo e mapeamento de classes gerados pelo train.py
    if not Path(ARQUIVO_MODELO).exists() or not Path(ARQUIVO_CLASSES).exists():
        raise SystemExit(f"'{ARQUIVO_MODELO}' não encontrado. Rode 'python train.py' primeiro.")
    modelo = keras.models.load_model(ARQUIVO_MODELO, compile=False)
    # tf.function compila a predição em grafo: ~7x mais rápido que chamar direto
    prever = tf.function(lambda x: modelo(x, training=False), reduce_retracing=True)
    with open(ARQUIVO_CLASSES, encoding="utf-8") as f:
        info = json.load(f)
    tamanho_img = info["tamanho_imagem"]
    classes = [info["classes"][str(i)] for i in range(len(info["classes"]))]

    # Carrega as imagens de cada gesto; quem não tiver imagem usa o emoji
    memes = {c["nome"]: carregar_referencia(c["nome"]) for c in classes}
    sem_imagem = [c for c in classes if memes[c["nome"]] is None and c["emoji"]]
    emojis = {}
    if sem_imagem:
        fonte = carregar_fonte_emoji()
        emojis = {c["nome"]: renderizar_emoji(c["emoji"], fonte, TAMANHO_EMOJI)
                  for c in sem_imagem}

    print("Gesto -> imagem exibida:")
    for c in classes:
        if memes[c["nome"]] is not None:
            origem = f"referencias/{c['nome']}"
        elif c["nome"] in emojis:
            origem = f"emoji {c['emoji']}"
        else:
            origem = "(nada)"
        print(f"  {c['nome']:15s} {origem}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a webcam.")

    historico = deque(maxlen=FRAMES_SUAVIZACAO)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espelha, igual ao coleta.py

        # Pré-processamento: BGR -> RGB e redimensiona (a normalização está no modelo)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        entrada = cv2.resize(rgb, (tamanho_img, tamanho_img)).astype(np.float32)
        probs = prever(entrada[np.newaxis]).numpy()[0]

        historico.append(probs)
        media = np.mean(historico, axis=0)
        indice = int(np.argmax(media))
        confianca = float(media[indice])
        nome = classes[indice]["nome"]

        if confianca >= args.limiar:
            painel = montar_painel(frame.shape[0], memes[nome], emojis.get(nome),
                                   f"{nome} {confianca:.0%}", (0, 255, 0))
        else:
            painel = montar_painel(frame.shape[0], None, None,
                                   f"? ({nome} {confianca:.0%})", (0, 165, 255))

        # Vídeo à esquerda, imagem do gesto à direita
        cv2.imshow("Gestos (Q para sair)", np.hstack([frame, painel]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
