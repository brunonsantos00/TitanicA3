import pandas as pd
import os

# Função para realizar engenharia de features nos dados
def feature_engineering(data):
    data = data.copy()  # Cria uma cópia dos dados originais
    data['Age'] = data['Age'].fillna(data['Age'].median())  # Preenche valores nulos na coluna 'Age' com a mediana
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())  # Preenche valores nulos na coluna 'Fare' com a mediana
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # Preenche valores nulos na coluna 'Embarked' com o valor mais frequente (moda)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Converte valores da coluna 'Sex' para valores numéricos
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # Converte valores da coluna 'Embarked' para valores numéricos
    data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)  # Remove colunas que não são necessárias para o modelo
    return data  # Retorna o DataFrame modificado

# Função para carregar e processar os dados
def load_and_process_data(raw_path, processed_path):
    # Carrega os dados brutos de treino e teste a partir dos arquivos CSV
    train_data = pd.read_csv(os.path.join(raw_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(raw_path, 'test.csv'))
    
    # Aplica a engenharia de features nos dados de treino e teste
    train_data_processed = feature_engineering(train_data)
    test_data_processed = feature_engineering(test_data)
    
    # Salva os dados processados em novos arquivos CSV
    train_data_processed.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
    test_data_processed.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)
    
    return train_data_processed, test_data_processed  # Retorna os dados processados de treino e teste

if __name__ == "__main__":
    base_path = 'C:\\Users\\Bruno\\Desktop\\tatanic_a3'  # Define o caminho base
    data_raw_path = os.path.join(base_path, 'data', 'raw')  # Define o caminho dos dados brutos
    data_processed_path = os.path.join(base_path, 'data', 'processed')  # Define o caminho dos dados processados
    
    os.makedirs(data_processed_path, exist_ok=True)  # Cria o diretório para os dados processados se ele não existir
    
    # Carrega e processa os dados
    train_data, test_data = load_and_process_data(data_raw_path, data_processed_path)
