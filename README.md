# MUAC Magic Hand Lab

Laboratório de interação visual que usa a webcam para formar um coração e letras com partículas controladas pelas mãos. O MediaPipe rastreia até duas mãos, o OpenCV exibe a imagem espelhada e o NumPy simula 2.000 partículas vermelhas.

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

O MUAC abre a webcam padrão no índice 0 pelo backend configurado em `CAMERA_BACKEND` (`DSHOW`, `MF` ou `AUTO`). A descoberta inicial e a ação **RESCAN** verificam os índices configurados em `CAMERA_PROBE_START` e `CAMERA_PROBE_END`. Um dispositivo só aparece se abrir e entregar um frame válido; a câmera ativa não é aberta novamente durante a sondagem. Como o OpenCV usado pelo projeto não enumera nomes confiáveis de dispositivos, a lista mostra `Camera N` com seu índice. Se um driver não aparecer com DirectShow, experimente `MF` (Media Foundation) em `config.py` e reinicie o aplicativo.

Pressione **C** ou abra **MUAC / VIDEO SOURCES** no canto inferior esquerdo. O painel lista apenas fontes verificadas, destaca a ativa e permite atualizar a lista com **RESCAN**. A sondagem ocorre em segundo plano; a lista é atualizada no loop principal. Se uma câmera escolhida falhar ao entregar vídeo, a câmera ativa anterior é preservada. Se usar a Iriun Webcam, abra o aplicativo no celular e o cliente no computador antes de atualizar a lista.

```powershell
.\.venv\Scripts\python.exe tests/probe_cameras.py
```

Para conferir o rastreamento e o esqueleto no preview independente, ajuste `CAMERA_INDEX` em `config.py` e execute o comando abaixo. O seletor do aplicativo vale para a sessão atual.

```powershell
.\.venv\Scripts\python.exe tests/preview_hands.py
```

## Gestos

O MUAC inicia com partículas livres. Levante o indicador para formar o coração ou mostre dois dedos para formar uma letra maiúscula de A a Z. O painel mostra o progresso enquanto você mantém o gesto.

| Gesto | O que acontece |
| --- | --- |
| Indicador levantado por 0,7 s | Seleciona HEART. A forma permanece quando a mão sai da câmera. |
| Dois dedos levantados por 0,7 s | Forma uma letra aleatória. Faça um V com indicador e médio e deixe o polegar recolhido. |
| Três dedos levantados por 2 s | Troca a letra ativa por outra. Levante indicador, médio e anelar com o polegar recolhido. Abaixe os dedos antes de pedir outra troca. |
| Cinco dedos abertos | A palma afasta apenas as partículas próximas. Ao afastar a mão, elas voltam à forma. |
| Pinça com polegar e indicador | Fora dos retângulos, captura e arrasta uma nuvem de partículas. Dentro de um retângulo, mantenha a pinça por `RECTANGLE_STYLE_SWITCH_HOLD` (0,45 s) para trocar apenas o preenchimento dessa figura; solte antes de outra troca. |
| Estalo com polegar e dedo médio | Dispersa as partículas uma vez e deixa a forma ativa se recompor. |
| Duas pinças simultâneas | O alvo é escolhido ao iniciar cada transformação: em AUTO, o ponto médio dentro de um retângulo bloqueado seleciona a camada superior; fora deles, seleciona as partículas. A distância ajusta a escala. Use **T** para forçar PARTICLES mesmo com o ponto médio sobre um retângulo, ou RECTANGLE para selecionar a figura que contém o ponto ou a mais próxima. Soltar mantém a transformação aplicada. |
| Polegar e indicador em L nas duas mãos | Aproxime as mãos para armar e depois afaste-as para criar um retângulo com preenchimento psicodélico. Abandone o gesto e forme o L novamente para criar outra figura; até `RECTANGLE_MAX_INSTANCES` retângulos independentes podem permanecer na tela. Duas pinças sobre um retângulo podem movê-lo/redimensioná-lo; reduzi-lo abaixo de 10% da largura e da altura da câmera por 0,30 s inicia o fade de exclusão individual. |

Cada retângulo possui geometria, transformação, estilo e preenchimento animado independentes, sem contorno cyan. Dentro de um retângulo, uma pinça sustentada por 0,45 s alterna PSYCHEDELIC → MONOCHROME → GLITCH; uma segunda pinça cancela a troca e prioriza TRANSFORM. Durante a confirmação, a pinça fica reservada à troca, sem capturar partículas nem mover a figura. Fora dos retângulos, uma pinça continua capturando partículas. Em sobreposição, a camada superior recebe a troca. O HUD mostra `RECTANGLE 02 / GLITCH` e o progresso da confirmação. MONOCHROME aplica plasma em tons de cinza somente no interior; GLITCH usa a própria imagem da webcam capturada na ROI antes de desenhar as partículas, distorce faixas e blocos, separa sutilmente os canais e aplica scanlines leves. O resultado distorcido mistura-se fortemente à ROI (`GLITCH_OPACITY`), formando uma janela preenchida sem alterar pixels externos. A textura psicodélica existente permanece inalterada. As figuras são renderizadas da mais antiga para a mais nova e a geometria permanece dentro do frame da câmera, respeitando `RECTANGLE_SCREEN_MARGIN`. O limite inicial é seis figuras simultâneas (`RECTANGLE_MAX_INSTANCES`).

Ao transformar HEART com duas pinças e alvo PARTICLES, manter a escala acima de `HEART_OVERLOAD_SCALE` por `HEART_OVERLOAD_HOLD_SECONDS` dispara uma explosão radial. O coração passa a partículas livres sem reinicializar posições; solte as pinças antes de iniciar outra transformação. Letras e transformações direcionadas ao retângulo não acionam overload. Os valores de limiar e força podem ser ajustados com `HEART_OVERLOAD_ENABLED`, `HEART_OVERLOAD_SCALE`, `HEART_OVERLOAD_HOLD_SECONDS` e `HEART_OVERLOAD_STRENGTH` em `config.py`.

Em `config.py`, `RECTANGLE_STYLE_SWITCH_HOLD` define a duração da pinça para trocar o preenchimento e `RECTANGLE_DEFAULT_FILL_STYLE` escolhe o estilo inicial de novos retângulos. `MONOCHROME_CONTRAST` ajusta o contraste do plasma em escala de cinza. `GLITCH_INTENSITY`, `GLITCH_BLOCK_SIZE`, `GLITCH_BAND_COUNT`, `GLITCH_RGB_SHIFT`, `GLITCH_SCANLINE_STRENGTH`, `GLITCH_OPACITY` e `GLITCH_SPEED` ajustam a força, as faixas, a separação de canais, as scanlines, a mistura e a velocidade das distorções. `PSYCHEDELIC_FILL_ENABLED` liga/desliga o interior, `PSYCHEDELIC_FILL_OPACITY` controla a mistura com a webcam, `PSYCHEDELIC_FILL_SPEED` ajusta a velocidade, `PSYCHEDELIC_FILL_SCALE` altera a frequência das ondas, `PSYCHEDELIC_FILL_SATURATION` ajusta a intensidade da paleta neon e `PSYCHEDELIC_FILL_HUE_SHIFT` gira suas cores em graus (0–360). `PSYCHEDELIC_FILL_MAX_DIMENSION` limita a resolução interna para preservar o desempenho.

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
| **T** | Alterna o alvo das duas pinças: AUTO → PARTICLES → RECTANGLE → AUTO. Em AUTO, o ponto médio seleciona o retângulo na camada superior quando estiver dentro dele; fora, seleciona as partículas livres, HEART ou letras. RECTANGLE escolhe a figura que contém o ponto médio ou a mais próxima. |
| **Q** ou **Esc** | Fecha o aplicativo. |

Durante uma transformação, o HUD mostra o alvo selecionado. Se o retângulo permanecer abaixo dos limites de exclusão, o HUD mostra `RECTANGLE / RELEASE TO CANCEL`; a exclusão ocorre após o tempo configurado em `RECTANGLE_DELETE_HOLD_SECONDS`, sem exigir que as pinças sejam soltas.

## Testes e demonstração

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

A suíte usa quadros e pontos sintéticos para verificar a orientação do coração, os gestos, as transições e a física. A precisão do estalo e o desempenho com vídeo ao vivo devem ser verificados com a webcam. [Roteiro de validação com Iriun](docs/reference-gap-analysis.md#validação-com-iriun) · [Imagem de orientação](docs/heart-orientation.png)

A demonstração de referência orienta a sequência do modo livre à formação do coração e à transformação com duas pinças. No MUAC, HEART e as letras se recompõem automaticamente após o estalo. [GIF de referência](https://github.com/HaiDuong1810/Gesture-Controlled-Particle-System/blob/main/assets/demo/demo_small.gif)

Veja também [a arquitetura do projeto](docs/architecture.md) e [a matriz de diferenças com a referência](docs/reference-gap-analysis.md).
