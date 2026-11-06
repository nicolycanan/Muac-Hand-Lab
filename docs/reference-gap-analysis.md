# Paridade funcional do MUAC Hand Lab

A referência examinada foi o [commit `cc1c251`](https://github.com/HaiDuong1810/Gesture-Controlled-Particle-System/tree/cc1c2519a6c7dd372bb3529168a09426f72d47c3). A árvore do commit não contém arquivo de licença e o README não declara uma. A implementação usa a geometria e os módulos do MUAC e não importa código nem imagens do outro projeto. Foram consultados integralmente `src/gesture_controller.py`, `src/particle_system.py`, `src/config.py`, `src/hand_tracker.py`, `src/shape_manager.py` e `main.py`.

## Matriz de diferenças

| Área | Referência | MUAC antes da V1 | MUAC V1 |
| --- | --- | --- | --- |
| `finger_is_extended()` | Distância punho–ponta relativa à distância punho–PIP | Ângulos de articulação e alcance longo em relação à palma | Combina ângulos PIP/DIP, alcance relativo e progressão ponta–punho. As medidas independem de escala e rotação no plano. |
| `get_shape_finger_count()` | Conta padrões de 1 a 4 dedos sem o polegar | Reconhecia indicador, mas a confirmação e a prioridade eram compartilhadas com os modos contínuos | O indicador é o único dos quatro dedos longos levantado. Mantém os 0,7 s de confirmação e emite SELECT_HEART uma vez. |
| `is_thumb_open()` e `is_full_open_hand()` | Abertura do polegar pela largura da palma, mais quatro dedos estendidos | Abertura do polegar por ângulo e distância do indicador | Usa ângulos e projeção do polegar para fora da palma. REPEL exige cinco dedos abertos. |
| `GestureController.process()` | Produz controles por mão, seleção mantida e evento de estalo. Duas pinças suspendem SNAP e seleção. | Havia gesto bruto e confirmação com um único tempo. SNAP ganhava de TRANSFORM. O efeito voltava a classificar a pinça. | Separa gesto bruto, candidato, confirmado, modo e evento. Prioriza TRANSFORM, BREAK, GRAB, REPEL e SELECT_HEART. Modos contínuos confirmam em 50 ms. A pinça usa os limiares 0,42 e 0,55 com o tamanho longitudinal da palma já fornecido pelo tracker. |
| `shape_manager.py` | Amostra pixels e cores de imagens de quatro presets | Máscara matemática com X e Y normalizados separadamente e amostragem sem pares simétricos | Preserva a proporção da curva e inverte Y uma vez na rasterização. Amostra estratos de área equivalente e espelha os pontos para simetria. HEART é o único preset. |
| `set_target()` | Atualiza destinos e cores, reduz velocidade e redefine a escala | Copiava destinos e sanitizava posições e velocidades | A geometria de origem fica somente leitura. Posições, velocidades, destinos e cores têm buffers separados. Transformar os destinos conserva posições e velocidades. |
| `update()` e REPEL | Repulsão radial, impulso da palma em movimento, amortecimento e colisão com limites | Havia repulsão local, com outros parâmetros e sem impulso da palma | Usa raio de 90 px, força 3,0 e contribuição de movimento 0,08 com queda linear pela distância. A mola atua continuamente em subpassos. O recorte na renderização descarta pixels fora da imagem. |
| `update()` e GRAB | Atrai partículas próximas a offsets aleatórios em uma nuvem de 22 px | Deslocava partículas próximas à pinça, enquanto a mola já as puxava de volta | Captura num raio de 180 px em torno do ponto médio entre os landmarks 4 e 8. Suspende a mola nas capturadas e mantém seus offsets. Os 22 px limitam a correção do centro da nuvem por quadro. Ao soltar, a nuvem retém parte do impulso e volta ao alvo. |
| `break_apart()` e loop de SNAP | Testa contato, separação e deslocamento do médio. Aplica impulso radial e muda para FREE. | Detectava separação sem expiração do contato e perdia o cooldown quando a mão sumia. A explosão usava ruído. | Contato a 0,48, soltura a 0,85, janela de 0,50 s e cooldown global de 0,8 s. Mede a velocidade do médio em relação à palma e usa o ponto de soltura para o impulso radial. O coração atual se reconstrói. |
| `move_shape()`, `scale_shape()`, `rotate_shape()` e loop de duas mãos | Suaviza deslocamentos, usa o meio das pinças como pivô de escala/rotação e corrige os limites da tela | Calculava transformações sem suavização. A escala visível parava em 1,15 e a rotação podia saltar. | Desloca pelo meio das pinças e transforma em torno do próprio centro da forma. Recalcula os destinos a partir da base imutável. Usa escala de 0,45 a 2,0, multiplicador de 0,97 a 1,03 por quadro e giro de até 4° por quadro, convertido para radianos. A linha das pinças tem período de 180°. Ao soltar ou perder uma mão, preserva a transformação e captura nova referência ao voltar. A forma pode sair parcialmente do quadro. |
| `draw()` e `src/config.py` | 2.000 partículas com raio 1 no código. Renderiza pixels vetorizados com cores dos assets. | 1.500 partículas, círculos antialias por partícula e halo | 2.000 partículas vermelhas, raio de 1 px e escrita vetorizada. |
| `src/hand_tracker.py` e captura em `main.py` | Câmera 0 a 640 × 480, MediaPipe com entrada de 256 × 192 e até duas mãos | Iriun por DirectShow, flip horizontal, BGR → RGB, dois conjuntos de landmarks e preview próprio | Preserva a entrada de 640 × 480, o tracker, a captura, o desenho e a tecla M. D exibe diagnóstico por mão e a decisão global. |

## Medição local

A 640 × 480, com 2.000 partículas, 40 quadros de aquecimento e 300 amostras, a mediana de REPEL + física + renderização caiu de 11,84 ms para 0,90 ms. O percentil 95 caiu de 12,92 ms para 1,15 ms. É uma medição isolada em memória, sem câmera, MediaPipe, HUD ou janela.

O diagnóstico de orientação usa a geometria matemática e testa a reentrância, os lóbulos, a ponta, a simetria e a transformação com giro zero. O pytest gera o PNG em seu diretório temporário. [Imagem de diagnóstico](heart-orientation.png).

## Validação com Iriun

1. Na raiz do MUAC, execute `.\.venv\Scripts\python.exe tests/preview_hands.py`. Mova uma mão e depois duas. Confira espelhamento, 21 pontos por mão e continuidade do esqueleto.
2. Execute `.\.venv\Scripts\python.exe main.py`. Pressione **D** para abrir o diagnóstico e **M** duas vezes para testar o esqueleto. Confira FPS e 2.000 partículas.
3. Pressione **H** para ocultar o coração. Mostre apenas o indicador por cerca de 0,7 s. Confira `RAW: INDEX_RAISED`, `CONFIRMED: INDEX_RAISED` e `ACTION: SELECT_HEART`. Retire a mão e confirme que o coração permanece com a ponta para baixo.
4. Abra os cinco dedos junto a uma borda do coração. Mova a palma e veja se só a região próxima reage. Retire a mão e verifique o reagrupamento.
5. Feche o punho e confira que não há ação. Faça uma pinça com polegar e indicador junto ao coração. Arraste a nuvem e solte. Confira `GRAB`, o deslocamento e o retorno das partículas.
6. Encoste o polegar no dedo médio e solte rapidamente. Confira um BREAK e a reconstrução. Repita dentro de 0,8 s e depois fora desse intervalo. Compare com uma soltura lenta.
7. Faça duas pinças. Desloque o meio, altere a distância e gire a linha das pinças ao mesmo tempo. Retire uma mão por um instante, retome a pinça e confira a continuidade sem salto. Solte as duas e confira que a forma fica no lugar.

O vídeo pode perder o contato ou a soltura do estalo entre quadros. Oclusão, desfoque, iluminação e o ângulo da mão também afetam a estimativa do dedo médio. Um teste sintético mede a lógica temporal, enquanto a precisão do gesto e o FPS de ponta a ponta dependem da validação ao vivo.
