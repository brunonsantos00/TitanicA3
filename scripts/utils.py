import pandas as pd

# Função para carregar dados a partir de um arquivo CSV
def load_data(file_path):
    return pd.read_csv(file_path)  # Carrega os dados e retorna um DataFrame
