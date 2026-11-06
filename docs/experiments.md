# Notas para experimentos

O `EffectManager` recebe o controlador de gestos após a confirmação e usa o resultado para escolher a interação. Um novo efeito pode usar essa mesma entrada e manter no próprio módulo o estado específico do visual.

1. Implemente o efeito em `effects/` com uma rotina para atualizar o estado e outra para desenhá-lo.
2. Integre a seleção ao `EffectManager` para usar o gesto confirmado e as coordenadas da mão.
3. Para um acionamento manual, use os atalhos de `MagicHandApp._handle_keys`.
4. Valide o comportamento com quadros sintéticos antes de calibrá-lo com a câmera.
