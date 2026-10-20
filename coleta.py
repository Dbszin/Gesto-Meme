"""
coleta.py - Captura fotos de treino pela webcam para um gesto.

Uso:
    python coleta.py joinha            # salva em dataset/joinha/
    python coleta.py neutro --qtd 300  # define quantas fotos capturar

Teclas:
    ESPAÇO -> inicia/pausa a captura automática
    Q      -> sai

Dica: durante a captura, mexa-se um pouco (distância, ângulo, mão
esquerda/direita, iluminação) para o modelo generalizar melhor.
"""

import argparse
import time
from pathlib import Path

import cv2

PASTA_DATASET = Path("dataset")
PASTA_REFERENCIAS = Path("referencias")
INTERVALO_SEG = 0.15  # tempo entre uma foto e outra na captura automática


def carregar_referencia(classe: str):
    """Procura o meme de referência do gesto (ex: referencias/joinha.jpeg)."""
    for ext in (".jpeg", ".jpg", ".png"):
        caminho = PASTA_REFERENCIAS / f"{classe}{ext}"
        if caminho.exists():
            img = cv2.imread(str(caminho))
            if img is not None:
                return cv2.resize(img, (160, 160))
    return None


def main():
    parser = argparse.ArgumentParser(description="Coleta de imagens de gestos")
    parser.add_argument("classe", help="Nome do gesto (vira o nome da pasta)")
    parser.add_argument("--qtd", type=int, default=200, help="Quantidade de fotos")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam")
    args = parser.parse_args()

    destino = PASTA_DATASET / args.classe
    destino.mkdir(parents=True, exist_ok=True)
    ja_existentes = len(list(destino.glob("*.jpg")))

    referencia = carregar_referencia(args.classe)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a webcam.")

    capturando = False
    salvas = 0
    ultimo = 0.0

    while salvas < args.qtd:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espelha, igual ao app.py

        # Salva o frame "limpo" (sem os textos desenhados por cima)
        if capturando and time.time() - ultimo >= INTERVALO_SEG:
            nome = destino / f"{args.classe}_{ja_existentes + salvas:05d}.jpg"
            cv2.imwrite(str(nome), frame)
            salvas += 1
            ultimo = time.time()

        # Interface
        tela = frame.copy()
        if referencia is not None:
            tela[10:170, -170:-10] = referencia
        status = "CAPTURANDO" if capturando else "PAUSADO (ESPACO para iniciar)"
        cor = (0, 0, 255) if capturando else (0, 200, 255)
        cv2.putText(tela, f"{args.classe}: {salvas}/{args.qtd}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(tela, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.imshow("Coleta de gestos", tela)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            capturando = not capturando
        elif tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{salvas} imagens salvas em {destino.resolve()}")


if __name__ == "__main__":
    main()
