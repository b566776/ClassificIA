
# Classificação de Obstáculos com Sensor Simulado e IA

## Planejamento do Projeto

### Ferramentas Principais
- Google Colab
- Bibliotecas Python: numpy, pandas, tensorflow/keras ou pytorch, scikit-learn, matplotlib
- Google AI Studio / API Gemini
- Git, GitHub

## Etapa 1: Geração de Dados Simulados (Mock Data)
1. **Objetivo:** Criar um conjunto de dados sintético que imite as leituras de distância de um sensor HC-SR04 para diversos cenários de obstáculos.
2. **Ambiente:** Google Colab Notebook
3. **Passo a Passo:**
   1. Abrir um novo Notebook no Google Colab: Nomear o notebook de forma descritiva (ex: 01_Geracao_Dados_Mock.ipynb).
   2. Instalar e Importar Bibliotecas: Importar numpy e pandas.
   3. Definir Tipos de Obstáculos: Criar uma lista ou dicionário com os tipos de obstáculos a serem simulados (ex: 'Pessoa', 'Carro', 'Objeto Pequeno', 'Parede Fixa'). Associar um ID numérico a cada tipo.
   4. Planejar Cenários de Simulação: Pensar em como cada obstáculo se comporta (ex: Pessoa andando - distância diminuindo/aumentando; Carro passando rápido - distância diminuindo/aumentando rapidamente; Objeto Pequeno - talvez só detectado perto ou com leituras mais erráticas; Parede Fixa - distância constante).
   5. Escrever Funções de Simulação: Criar funções Python que gerem sequências de leituras de distância ao longo do tempo para cada tipo de obstáculo.
   6. Incluir parâmetros como distância inicial, velocidade (positiva ou negativa para aproximar/afastar), duração do evento.
   7. Adicionar uma pequena variação aleatória (ruído) para simular as imperfeições do sensor.
   8. Gerar o Dataset Completo: Rodar as funções de simulação para gerar várias sequências de dados para cada tipo de obstáculo. Combinar essas sequências em um DataFrame Pandas.
   9. Cada linha do DataFrame pode representar um instante de tempo, com colunas para:
      - timestamp
      - distancia_cm
      - tipo_obstaculo (label numérico ou string)
      - id_cenario (para agrupar sequências do mesmo evento)
   10. (Opcional) Explorar com Gemini: Utilizar a API Gemini (se configurada no Colab) ou copiar trechos dos dados gerados para o Google AI Studio para pedir ao modelo que descreva textualmente o que "parece" estar acontecendo em um determinado cenário de dados simulados. Isso pode ajudar a validar se a simulação faz sentido.
   11. Salvar o Dataset : Salvar o DataFrame em um formato de arquivo, como CSV (.csv) ou HDF5 (.h5), de preferência no Google Drive montado no Colab para fácil acesso em outros notebooks. Ex: dados_sensores_simulados.csv.
   12. Documentar o Notebook: Adicionar comentários explicando o código e células de texto (Markdown) descrevendo os tipos de obstáculos, parâmetros de simulação e a estrutura do dataset gerado.

## Etapa 2: Pré-processamento de Dados e Treinamento do Modelo de IA
1. **Objetivo:** Preparar os dados simulados e treinar um modelo de Machine Learning para classificar as sequências de leituras do sensor.
2. **Ambiente:** Google Colab Notebook
3. **Passo a Passo:**
   1. Abrir um novo Notebook no Google Colab: Nomear (ex: 02_Treinamento_Modelo_IA.ipynb).
   2. Instalar e Importar Bibliotecas: Importar pandas, numpy, bibliotecas de ML (tensorflow e keras OU pytorch), sklearn.model_selection para split.
   3. Carregar os Dados: Carregar o arquivo de dados simulados salvo na Etapa 1 (ex: dados_sensores_simulados.csv) para um DataFrame Pandas.
   4. Pré-processar os Dados:
      - Agrupar os dados por id_cenario para obter as sequências completas de cada evento simulado.
      - Estruturar os dados em sequências de comprimento fixo (janelamento): Transformar as sequências de leituras em amostras de entrada para o modelo (cada amostra é uma janela de N leituras). Definir o tamanho da janela N.
      - Associar o rótulo (tipo de obstáculo) a cada sequência/janela.
      - Normalizar ou escalar os dados de distância (ex: para o intervalo [0, 1]).
   5. Dividir o Dataset: Separar o conjunto de dados processado em conjuntos de treinamento, validação e teste (ex: 70% treino, 15% validação, 15% teste) usando sklearn.model_selection.train_test_split.
   6. Construir o Modelo de IA:
      - Escolher uma arquitetura adequada para dados sequenciais (ex: Camada LSTM, GRU, ou Conv1D em Keras/PyTorch).
      - Definir a estrutura do modelo: Camada(s) de entrada com o formato da sequência (N leituras), camada(s) recorrentes/convolucionais, camadas densas (Dense), e uma camada de saída com ativação 'softmax' (para classificação multiclasse) com o número de neurônios igual ao número de tipos de obstáculos.
      - Compilar o modelo: Definir a função de perda (ex: 'categorical_crossentropy'), o otimizador (ex: 'adam') e as métricas (ex: 'accuracy').
   7. Treinar o Modelo:
      - Treinar o modelo usando os dados de treinamento e validação.
      - Configurar callbacks úteis (ex: Early Stopping para parar o treino quando a validação para de melhorar, Model Checkpoint para salvar o melhor modelo).
      - Rodar o treinamento (aproveitando a GPU/TPU do Colab).
   8. Salvar o Modelo Treinado: Salvar o modelo treinado (pesos e arquitetura) em um arquivo (ex: formato HDF5 .h5 para Keras, ou .pt/.pth para PyTorch), novamente preferencialmente no Google Drive.
   9. Documentar o Notebook: Explicar as etapas de pré-processamento, a arquitetura do modelo escolhido, os hiperparâmetros de treinamento e o processo de treino.

## Etapa 3: Avaliação e Análise do Modelo
1. **Objetivo:** Medir o desempenho do modelo treinado e analisar seus resultados, identificando pontos fortes e fracos.
2. **Ambiente:** Google Colab Notebook e Google AI Studio (uso complementar)
3. **Passo a Passo:**
   1. Abrir um novo Notebook no Google Colab: Nomear (ex: 03_Avaliacao_Analise.ipynb).
   2. Instalar e Importar Bibliotecas: Importar pandas, numpy, a biblioteca de ML usada (tensorflow ou pytorch), sklearn.metrics, matplotlib, seaborn. Opcional: google.generativeai se for usar a API Gemini diretamente no Colab.
   3. Carregar o Modelo e os Dados de Teste: Carregar o modelo treinado salvo na Etapa 2 e o conjunto de dados de teste preparado na Etapa 2.
   4. Realizar Previsões: Usar o modelo carregado para fazer previsões sobre os dados de teste.
   5. Calcular Métricas de Avaliação: Usar sklearn.metrics para calcular métricas relevantes para classificação (acurácia, precisão por classe, recall por classe, F1-score por classe, matriz de confusão).
   6. Visualizar Resultados:
      - Gerar um mapa de calor da matriz de confusão usando seaborn para visualizar onde o modelo acertou e errou.
      - Plotar exemplos de sequências de teste com as previsões do modelo versus os rótulos reais.
   7. Analisar Erros: Identificar os tipos de obstáculos que o modelo teve mais dificuldade em classificar (examinando a matriz de confusão e exemplos de erros).
   8. (Opcional) Utilizar Google AI Studio / Gemini para Análise Textual:
      - Copiar as métricas de avaliação calculadas (acurácia, matriz de confusão) e colar no Google AI Studio.
      - Criar um prompt pedindo ao Gemini para analisar essas métricas e explicar em linguagem natural o desempenho do modelo, os principais pontos de acerto e os tipos de erros mais comuns.
      - Você também pode copiar exemplos de sequências onde o modelo errou e pedir ao Gemini para "especular" por que aquela sequência específica pode ter sido confusa para a IA, com base na variação dos números.
   9. Documentar o Notebook: Registrar as métricas de avaliação, incluir as visualizações e descrever as conclusões da análise (quais obstáculos são bem classificados, quais são problemáticos, possíveis razões).

## Etapa 4: Publicação no GitHub
1. **Objetivo:** Criar um repositório público no GitHub para hospedar o código do projeto, tornando-o acessível e compartilhável.
2. **Ferramentas:** Navegador Web (para GitHub), Git (linha de comando ou cliente gráfico)
3. **Passo a Passo:**
   1. Criar Conta GitHub: Se ainda não tiver, acesse github.com e crie uma conta gratuita.
   2. Criar Novo Repositório: Clique no botão '+' no canto superior direito do GitHub e selecione "New repository".
   3. Dê um nome claro e descritivo ao repositório (ex: Obstacle_Classifier_Simulated_Sensor_IA_Colab).
   4. Adicione uma breve descrição do projeto.
   5. Escolha "Public" para que seja acessível a todos.
   6. Marque a opção "Add a README file" (você irá editá-lo depois).
   7. (Opcional) Adicionar um .gitignore (escolher o template Python é um bom começo) e uma licença.
   8. Clique em "Create repository".
   9. Escrever/Editar o README: No repositório criado, edite o arquivo README.md.
      - Inclua um título claro do projeto.
      - Escreva uma descrição detalhada: O que o projeto faz (classificação de obstáculos simulados com IA), as tecnologias usadas (Python, Colab, TensorFlow/PyTorch, Gemini complementar), o escopo (simulação, sem hardware/GCP direto).
      - Descreva a estrutura do repositório (quais arquivos/notebooks estão presentes).
      - Forneça instruções sobre como rodar o projeto: Como abrir os notebooks no Google Colab, qual a ordem das etapas (1, 2, 3), o que esperar.
      - Mencione o uso complementar do Google AI Studio/Gemini para análise.
      - Incluir capturas de tela (opcional, mas útil).
   10. Organizar Arquivos Localmente: No seu computador (ou diretamente no Google Drive onde os notebooks foram salvos), organize os arquivos do projeto: os notebooks Colab (.ipynb), o arquivo de dados simulados (se não for muito grande, caso contrário, descrever como gerar no README), o modelo treinado (se não for muito grande, caso contrário, descrever como retreinar), e o arquivo README.md e .gitignore (se adicionados no GitHub).
   11. Inicializar Repositório Git Local (ou Usar a Interface Web do GitHub):
       - Opção Recomendada (linha de comando): Abra um terminal na pasta local do seu projeto. Rode git init.
       - Adicionar Arquivos: Rode git add . para adicionar todos os arquivos.
       - Commit Inicial: Rode git commit -m "Initial project commit - Simulated Obstacle Classifier" (ou uma mensagem descritiva).
       - Conectar ao Repositório Remoto: No GitHub, na página principal do seu repositório, copie a URL (HTTPS ou SSH). Volte ao terminal e rode git remote add origin <URL_do_seu_repositorio>.
       - Enviar para o GitHub: Rode git push -u origin main (ou master, dependendo da branch padrão). Pode ser que você precise autenticar.
   12. (Opção Alternativa) Interface Web: Para projetos pequenos, você pode simplesmente usar a interface do GitHub para fazer upload dos arquivos um por um ou arrastar e soltar pastas.
   13. Verificar Publicação: Vá para a página do seu repositório no GitHub e confirme que todos os arquivos estão presentes. Verifique se o README está sendo exibido corretamente.
   14. Compartilhar o Link: O link para o seu repositório agora é público e pode ser compartilhado (ex: https://github.com/SeuUsuario/NomeDoSeuRepositorio).

---
Este plano fornece um roteiro claro, aproveitando as ferramentas especificadas (Colab para o core ML, Gemini/AI Studio para apoio e análise textual, evitando GCP) e garantindo que o resultado do trabalho seja compartilhado e visível através do GitHub. Boa sorte com o projeto na imersão!