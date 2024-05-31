import pandas as pd
import numpy as np
import os

# Função para realizar engenharia de features nos dados
def feature_engineering(data):
    data = data.copy()  # Cria uma cópia dos dados para evitar modificar o original
    data['Age'] = data['Age'].fillna(data['Age'].median())  # Preenche valores nulos na coluna 'Age' com a mediana
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())  # Preenche valores nulos na coluna 'Fare' com a mediana
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # Preenche valores nulos na coluna 'Embarked' com o valor mais frequente (moda)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Converte a coluna 'Sex' para valores numéricos
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # Converte a coluna 'Embarked' para valores numéricos
    
    # Criação de novas features
    data['Family_Size'] = data['SibSp'] + data['Parch']  # Cria a feature 'Family_Size' somando 'SibSp' (irmãos/cônjuges) e 'Parch' (pais/filhos)
    data['IsAlone'] = (data['Family_Size'] == 0).astype(int)  # Cria a feature 'IsAlone' que indica se a pessoa está viajando sozinha
    data['Fare_Per_Person'] = data['Fare'] / (data['Family_Size'] + 1)  # Cria a feature 'Fare_Per_Person' dividindo 'Fare' pelo tamanho da família + 1
    
    # Remove colunas desnecessárias para o modelo
    data.drop(['Cabin', 'Ticket', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)
    return data

# Função para carregar e processar os dados
def load_and_process_data(raw_path, processed_path):
    # Carrega os dados brutos
    train_data = pd.read_csv(os.path.join(raw_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(raw_path, 'test.csv'))
    
    # Aplica engenharia de features nos dados de treino e teste
    train_data_processed = feature_engineering(train_data)
    test_data_processed = feature_engineering(test_data)
    
    # Salva os dados processados
    train_data_processed.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
    test_data_processed.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)
    
    return train_data_processed, test_data_processed

if __name__ == "__main__":
    base_path = r'C:\Users\Bruno\Desktop\tatanic_a3'
    data_raw_path = os.path.join(base_path, 'data', 'raw')
    data_processed_path = os.path.join(base_path, 'data', 'processed')
    
    # Cria o diretório para os dados processados se não existir
    os.makedirs(data_processed_path, exist_ok=True)
    
    # Carrega e processa os dados
    train_data, test_data = load_and_process_data(data_raw_path, data_processed_path)
