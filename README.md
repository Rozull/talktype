# talktype

Ditado por voz local para qualquer aplicativo do Windows.
Segure o **Ctrl direito**, fale depois do sinal e solte: o texto aparece onde está o cursor —
no chat, no navegador, na IDE, no terminal ou no e-mail.

Tudo roda na sua máquina. O reconhecimento de fala (faster-whisper), a revisão opcional por IA
(Ollama) e todos os dados ficam no computador. O áudio nunca é gravado em disco, e a única
conexão de rede é o download do modelo na primeira execução.

## Instalação

Requisitos: Windows 10 ou 11 e internet na instalação. Nada precisa de administrador.

1. Baixe o **`talktype-setup.exe`** da [página de versões](https://github.com/Rozull/talktype/releases/latest).
2. Abra o arquivo. Como o instalador não é assinado, o Windows pode mostrar "O Windows protegeu o
   computador": clique em **Mais informações** e em **Executar assim mesmo**.
3. Uma janela mostra o progresso. Na primeira vez são baixados o Python, as dependências
   (cerca de 470 MB instalados) e, com uma GPU NVIDIA, as bibliotecas CUDA (cerca de 2 GB). No
   fim, o talktype abre sozinho.

Também dá para instalar pelo PowerShell, sem o `.exe`:

```powershell
$f = "$env:TEMP\talktype-setup.ps1"; irm https://github.com/Rozull/talktype/releases/latest/download/setup.ps1 -OutFile $f; powershell -NoProfile -ExecutionPolicy Bypass -File $f
```

O que o instalador faz:

- Coloca o app em `%LOCALAPPDATA%\talktype\app`, cria o atalho **talktype** no menu Iniciar e
  registra o talktype em **Configurações → Aplicativos instalados**, de onde ele é desinstalado.
- **Com GPU NVIDIA**, usa o modelo `large-v3-turbo` na GPU. **Sem GPU NVIDIA**, não baixa as
  bibliotecas CUDA e configura o modelo `parakeet-v3`, o mais rápido na CPU (menos de 1,5 s
  por ditado nos testes). O modelo de voz é baixado na primeira execução (0,7 a 1,6 GB).
- Para **atualizar**, rode o instalador de novo: ele fecha o talktype, troca o app pela versão
  mais recente e abre de novo. A configuração e o histórico (`%APPDATA%\talktype`) nunca são
  alterados.

Para **desinstalar**: Configurações → Aplicativos instalados → talktype → Desinstalar. A
configuração e o histórico ficam em `%APPDATA%\talktype`, para uma reinstalação futura. Para
apagá-los também, rode `scripts\uninstall.ps1 -Purge` na pasta do app.

Opcional: o [Ollama](https://ollama.com/) com `qwen2.5:7b-instruct`, para a revisão por IA.

### Instalação para desenvolvimento

Com [uv](https://docs.astral.sh/uv/) e Git:

```powershell
git clone https://github.com/Rozull/talktype.git
cd talktype
uv sync
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install.ps1
```

O `uv sync` instala também as ferramentas de teste e as bibliotecas CUDA (os grupos `dev` e
`gpu`). O `install.ps1` cria o atalho **talktype** apontando para este clone. Ele pode ser
executado de novo sem problemas e nunca ativa a inicialização automática, que é feita pela
bandeja. Para atualizar: `git pull` e `uv sync`.

## Primeira execução

Abra **talktype** no menu Iniciar (ou rode `uv run talktype` para ver o log no terminal).

1. O ícone aparece na bandeja do sistema. Na primeira vez, o modelo de voz é baixado do
   Hugging Face (1,6 GB para o `large-v3-turbo`, 0,7 GB para o `parakeet-v3`), com o progresso
   na bandeja.
2. Quando a bandeja mostrar **Pronto**, um aviso único explica o gesto de ditado.
3. Com o modelo já baixado, o talktype fica pronto em poucos segundos.

Se o download falhar (sem conexão, proxy, disco cheio), a bandeja mostra o motivo e a opção
**Tentar baixar novamente**. Só uma instância roda por usuário; abrir de novo apenas avisa que o
talktype já está em execução.

## Como usar

| Gesto | Resultado |
|---|---|
| Segurar o Ctrl direito por 1 s | Toca o som de início e aparece o painel: fale. Solte para inserir o texto. |
| Tocar o Ctrl direito duas vezes (em até 400 ms) | Modo mãos livres: fale à vontade e toque o Ctrl direito uma vez para terminar. |
| Esc durante a gravação | Cancela: nada é transcrito, inserido ou guardado. |
| Ctrl+Alt+Shift+V | Reinsere o último ditado (funciona até com o ditado pausado). |

- Toques rápidos no Ctrl direito e atalhos como Ctrl direito+C continuam funcionando normalmente.
- O painel mostra o nível do microfone, uma prévia do texto e, no fim, o resultado.
- Diga "nova linha" ou "novo parágrafo" para quebrar linhas. Quebras de linha nunca enviam
  mensagens: no modo colar vão dentro do texto e no modo digitar viram Shift+Enter.
- Se não for possível inserir (janela de administrador, nenhum campo com foco), o texto fica
  no clipboard (Ctrl+V) e no **Histórico** da bandeja.
- Gravações param sozinhas em 5 minutos; o tempo restante aparece nos últimos 30 segundos.

A bandeja também tem: modo colar/digitar, prévia ao vivo, limpeza, comandos falados, revisão
por IA, pausar, iniciar com o Windows, escolha do microfone, histórico (copiar ou inserir de
novo), abrir e recarregar a configuração, exportar e importar.

## Configuração

O arquivo fica em `%APPDATA%\talktype\config.toml` (ou em `%TALKTYPE_HOME%\config.toml`, se essa
variável estiver definida). Ele é criado na primeira execução com todas as opções comentadas —
veja também [`config.example.toml`](config.example.toml). Na mesma pasta ficam `history.json`,
`state.json` e os logs (`logs\talktype.log`, que nunca contêm o texto ditado).

Edite o arquivo e use **Recarregar config** na bandeja. Um valor inválido nunca é aplicado: o
valor anterior continua valendo e o aviso diz qual chave e por quê. Itens comuns:

- `asr.vocabulary`: nomes e termos que devem sair com a grafia certa.
- `asr.model`: `large-v3-turbo` (padrão), `large-v3`, `medium`, `small` ou `parakeet-v3`.
- `commands.items`: frases faladas que viram texto.
- `cleanup.*`: remoção de hesitações ("hum", "ahn"...), substituições e o ponto final
  automático (`final_period`).
- `trigger.*`: tecla e tempos do gesto.
- `history.repaste_hotkey`: o atalho de reinserção.

Mudanças feitas durante um ditado valem a partir do ditado seguinte. **Exportar config** gera
um arquivo para levar a outra máquina; **Importar config** valida tudo, guarda um backup da
configuração atual e aplica a nova de uma vez.

## Revisão por IA (opcional)

Com o [Ollama](https://ollama.com/) instalado:

```powershell
ollama pull qwen2.5:7b-instruct
```

Ative **Revisão por IA** na bandeja (ou `llm.enabled = true`). O texto é corrigido segundo a
instrução em `llm.instruction`, sem mudar o sentido. Só endereços locais (`127.0.0.1`,
`localhost` ou `::1`) são aceitos. Se o servidor não responder a tempo (5 s), o texto sem revisão
é inserido com o aviso "revisão por IA ignorada". Também funciona com servidores compatíveis com
a API da OpenAI (`llm.provider = "openai_compatible"`).

## Desenvolvimento e testes

O portão de qualidade de cada mudança:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

O `pytest` padrão roda os testes `unit` e `integration`. Os testes de integração usam o
teclado, o clipboard e o foco de verdade: não mexa no computador enquanto rodam.

As suítes `gpu` (modelos reais) e `e2e` (o app completo, dirigido por teclas injetadas e áudios
de teste) rodam só nesta máquina, com a área de trabalho desbloqueada, a GPU e os modelos já
baixados; os casos de revisão por IA precisam do Ollama rodando:

```powershell
uv run pytest -m "gpu or e2e"
```

Dicas:

- Os testes de revisão por IA deixam o modelo do Ollama carregado na GPU por alguns minutos
  (cerca de 4,7 GB). Com pouca memória livre, o talktype escolhe a CPU e o IT-035 falha. Antes
  de `-m gpu`, rode `ollama stop qwen2.5:7b-instruct`.
- Um clique ou uma tecla durante os testes tira o foco da janela de teste. Os testes de injeção
  falham com "focus stolen by …" e os de ditado deixam de confirmar a gravação: rode de novo com
  o computador parado.
- O IT-011 (alvo elevado) é pulado por padrão. Para rodá-lo, abra o terminal sem privilégios
  elevados, defina `$env:TALKTYPE_TEST_ELEVATED = "1"` e aceite o UAC do Bloco de Notas.

Os áudios de teste ficam em `tests/fixtures/audio` e podem ser recriados com
`scripts\gen_fixtures.ps1` (voz "Microsoft Maria Desktop").

### Publicar uma versão

1. Atualize `version` no `pyproject.toml` e `__version__` em `src/talktype/__init__.py`, e
   rode `uv lock`.
2. Gere o instalador (cria `dist\talktype-setup.exe`, cerca de 200 KB):
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_setup.ps1`
3. Crie a tag e a versão no GitHub com os dois arquivos:
   `gh release create vX.Y.Z dist\talktype-setup.exe scripts\setup.ps1 --generate-notes`.
   O instalador baixa o código da tag mais recente, por isso o repositório precisa ser público.
