# Arquitetura do MUAC Magic Hand Lab

O loop em `Magic_hand.py` captura um quadro, espelha o vídeo e envia a imagem ao rastreador. As camadas seguintes usam as coordenadas dos pontos no mesmo sistema de referência da imagem.

- `core/` abre a câmera, extrai os pontos das mãos, confirma gestos ao longo do tempo e reúne o estado do aplicativo.
- `effects/` escolhe qual interação aplicar. A transformação do coração usa a geometria de origem para atualizar os destinos.
- `particles/` gera a forma matemática e atualiza posições e velocidades. A geometria de origem fica somente leitura.
- `rendering/` desenha sobre a imagem o painel de estado e o diagnóstico de gestos.
- `tests/` verifica as transições, a física, o desenho e o loop de exibição com dados sintéticos.
