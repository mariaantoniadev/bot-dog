Projeto Bot Dog
Este projeto tem como intuito demonstrar os conhecimentos na matéria de Inteligência Artificial e fiscalizar/detectar na câmera quando e se meu cachorro está na peça.
Tive esta idéia pois meu cachorro come muitas coisas que não devia e estraga muitos móveis, gostaria nas câmeras instaladas aqui em casa poder fazer essa detecção e ter mais controle sobre ele.

Passos para criar e ativar um ambiente virtual:
Criar o ambiente virtual:
python -m venv env-visao
Ativar o ambiente virtual:
.\env-visao\Scripts\activate

Instalação de Dependências
Instale as dependências listadas no arquivo requirements.txt:
pip install -r requirements.txt
Conteúdo do arquivo requirements.txt:
tensorflow
opencv-python
numpy
pillow
scipy
matplotlib

Desativação do Ambiente Virtual 
deactivate

Neste projeto usei esses requirements, tive alguns problemas na detecção do animal na câmera mas em alguns testes consegui os resultados corretos.

Deixei anexado o print do projeto funcionando nas imagens: test1  e test2

Ainda não consegui realizar total precisão, acredito que as fotos que escolhi nao são as melhores, mas estou feliz em realizar o projeto.