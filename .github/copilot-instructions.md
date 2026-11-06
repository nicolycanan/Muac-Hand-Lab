# MUAC — Git Workflow

Este projeto utiliza um fluxo de desenvolvimento controlado pela pessoa responsável pelo repositório.

**Regras obrigatórias:**

1. O Agent pode analisar, implementar, modificar e testar código localmente.
2. Após cada implementação, deve apresentar um resumo das alterações.
3. Deve informar quais arquivos foram modificados e quais testes foram executados.
4. Deve perguntar explicitamente:

   "A implementação foi aprovada? Posso criar o commit e enviar as alterações para a branch main do GitHub?"

5. Nunca executar `git push` sem uma resposta afirmativa explícita para a publicação atual.
6. Nunca interpretar a solicitação de implementação como autorização automática para publicar.
7. Se a implementação não for aprovada, continuar ajustando o código localmente.
8. Não executar `git push --force`.
9. Não descartar alterações locais sem autorização.
10. Não misturar funcionalidades não relacionadas em um mesmo commit.
11. Preferir commits pequenos, com mensagens claras e descritivas.
12. Executar os testes relevantes antes de propor a publicação.
13. Caso existam conflitos com `origin/main`, interromper a publicação, explicar o conflito e solicitar orientação antes de resolvê-lo.
14. Após um push autorizado, confirmar o commit publicado e a branch de destino.
15. Se houver alguma dúvida sobre a autorização, não publicar.

**Fluxo esperado:**

Implementação → Testes → Resumo → Aprovação → Commit → Push → Confirmação.

Essas instruções devem ser seguidas em todas as futuras tarefas do Agent neste repositório.
