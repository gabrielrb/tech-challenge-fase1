# Mesma versão do Python do Google Colab (3.13)
FROM python:3.13-slim
# Pasta de trabalho dentro do contêiner
WORKDIR /app
# Copia a lista de bibliotecas
COPY requirements.txt .
# Instala as bibliotecas (o TensorFlow deixa este passo demorado)
RUN pip install --no-cache-dir -r requirements.txt
# Copia os notebooks, o data.csv e o resto do repositório
COPY . .
# Porta do Jupyter
EXPOSE 8888
# Abre o Jupyter Lab sem senha (uso local ou no GitHub Codespaces)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token="]
