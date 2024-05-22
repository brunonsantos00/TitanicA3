import pandas as pd
import os
import joblib

# Função para carregar o modelo e o scaler
def load_model_and_scaler(models_path):
    model = joblib.load(os.path.join(models_path, 'model.pkl'))  # Carrega o modelo treinado
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))  # Carrega o scaler
    return model, scaler  # Retorna o modelo e o scaler

# Função para fazer previsões e criar o arquivo de submissão
def make_predictions(test_data, model, scaler, submissions_path):
    passenger_ids = test_data['PassengerId']  # Armazena os IDs dos passageiros
    
    # Remove a coluna PassengerId e garante que as colunas correspondam
    features = test_data.drop('PassengerId', axis=1)
    test_data_scaled = scaler.transform(features)  # Aplica o scaler nas features
    
    test_data['Survived'] = model.predict(test_data_scaled)  # Faz previsões
    test_data['PassengerId'] = passenger_ids  # Adiciona de volta o PassengerId
    
    # Cria o arquivo de submissão
    submission = test_data[['PassengerId', 'Survived']]
    submission.to_csv(os.path.join(submissions_path, 'submission.csv'), index=False)  # Salva o arquivo de submissão

if __name__ == "__main__":
    base_path = 'C:\\Users\\Bruno\\Desktop\\tatanic_a3'  # Define o caminho base
    data_processed_path = os.path.join(base_path, 'data', 'processed')  # Define o caminho dos dados processados
    models_path = os.path.join(base_path, 'models')  # Define o caminho dos modelos
    submissions_path = os.path.join(base_path, 'submissions')  # Define o caminho das submissões

    os.makedirs(submissions_path, exist_ok=True)  # Cria o diretório de submissões se ele não existir

    test_data = pd.read_csv(os.path.join(data_processed_path, 'test_processed.csv'))  # Carrega os dados de teste processados
    model, scaler = load_model_and_scaler(models_path)  # Carrega o modelo e o scaler
    make_predictions(test_data, model, scaler, submissions_path)  # Faz previsões e cria o arquivo de submissão
