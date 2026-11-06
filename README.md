# MUAC Magic Hand Lab

Laboratório de interação visual que usa a webcam para controlar um coração de partículas com as mãos. O MediaPipe rastreia até duas mãos, o OpenCV exibe a imagem espelhada e o NumPy simula 2.000 partículas vermelhas.

## Instalação

No Windows, abra o PowerShell na pasta do projeto. Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Inicie o aplicativo com o comando abaixo.

```powershell
.\.venv\Scripts\python.exe main.py
```

## Configuração da câmera

O MUAC abre a webcam padrão no índice 0 pelo DirectShow. Para trocar, clique em **CAMERA** no canto inferior esquerdo da janela ou pressione **C** para abrir o seletor. Clique no índice desejado. Se o dispositivo não entregar vídeo, o MUAC informa que a câmera está indisponível e mantém o vídeo da câmera ativa.

Se usar a Iriun Webcam, abra o aplicativo no celular e o cliente no computador antes de escolher o índice. A ordem dos índices é definida pelo Windows. O seletor exibe índices, e o seguinte comando ajuda a identificar quais entregam vídeo.

```powershell
.\.venv\Scripts\python.exe tests/probe_cameras.py
```

Para conferir o rastreamento e o esqueleto no preview independente, ajuste `CAMERA_INDEX` em `config.py` e execute o comando abaixo. O seletor do aplicativo vale para a sessão atual.

```powershell
.\.venv\Scripts\python.exe tests/preview_hands.py
```

## Gestos

O MUAC inicia com partículas livres. Levante o indicador e acompanhe a confirmação no painel. Ao confirmar o gesto, as partículas se reúnem no coração.

| Gesto | O que acontece |
| --- | --- |
| Indicador levantado por 0,7 s | Seleciona HEART. A forma permanece quando a mão sai da câmera. |
| Cinco dedos abertos | A palma afasta apenas as partículas próximas. Ao afastar a mão, elas voltam à forma. |
| Pinça com polegar e indicador | Captura e arrasta uma nuvem de partículas. Ao soltar, elas voltam ao coração. |
| Estalo com polegar e dedo médio | Dispersa as partículas uma vez e deixa o coração se recompor. |
| Duas pinças simultâneas | O ponto médio desloca o coração, a distância ajusta a escala e a inclinação gira a forma. Uma linha marca os pontos de pinça e exibe a escala e o ângulo. Soltar mantém a transformação aplicada. |

## Teclado

| Tecla | Ação |
| --- | --- |
| **C** | Abre ou fecha o seletor de câmera. |
| **H** | Alterna entre partículas livres e HEART. |
| **R** | Espalha e reagrupa as partículas quando HEART está ativo. |
| **E** | Dispara a dispersão radial. |
| **G** | Ativa ou desativa o reconhecimento de gestos. |
| **D** | Mostra o diagnóstico por mão, o gesto confirmado, o modo e a ação. |
| **M** | Mostra ou oculta o esqueleto das mãos. |
| **Q** ou **Esc** | Fecha o aplicativo. |

## Testes e demonstração

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

A suíte usa quadros e pontos sintéticos para verificar a orientação do coração, os gestos, as transições e a física. A precisão do estalo e o desempenho com vídeo ao vivo devem ser verificados com a webcam. [Roteiro de validação com Iriun](docs/reference-gap-analysis.md#validação-com-iriun) · [Imagem de orientação](docs/heart-orientation.png)

A demonstração de referência orienta a sequência do modo livre à formação do coração e à transformação com duas pinças. No MUAC, HEART é a única forma disponível e se recompõe automaticamente após o estalo. [GIF de referência](https://github.com/HaiDuong1810/Gesture-Controlled-Particle-System/blob/main/assets/demo/demo_small.gif)

Veja também [a arquitetura do projeto](docs/architecture.md) e [a matriz de diferenças com a referência](docs/reference-gap-analysis.md).
