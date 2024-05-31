import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
import joblib
from tf_keras.models import load_model

# Configurar o uso do Keras legado
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Função para carregar o modelo e o scaler
def load_model_and_scaler(models_path):
    model = load_model(os.path.join(models_path, 'model.h5'))  # Carrega o modelo treinado
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))  # Carrega o scaler salvo
    return model, scaler

# Função para fazer predições nos dados de teste
def make_predictions(test_data, model, scaler, submissions_path):
    passenger_ids = test_data['PassengerId']  # Salva os IDs dos passageiros
    
    features = test_data.drop('PassengerId', axis=1)  # Remove a coluna 'PassengerId' para fazer predições
    test_data_scaled = scaler.transform(features)  # Aplica o scaler aos dados de teste
    
    predictions = (model.predict(test_data_scaled) > 0.5).astype(int)  # Faz predições e converte para binário
    test_data['Survived'] = predictions  # Adiciona a coluna 'Survived' com as predições
    test_data['PassengerId'] = passenger_ids  # Adiciona a coluna 'PassengerId' novamente
    
    submission = test_data[['PassengerId', 'Survived']]  # Cria o DataFrame de submissão
    submission.to_csv(os.path.join(submissions_path, 'submission.csv'), index=False)  # Salva o arquivo de submissão

if __name__ == "__main__":
    base_path = r'C:\Users\Bruno\Desktop\tatanic_a3'
    data_processed_path = os.path.join(base_path, 'data', 'processed')
    models_path = os.path.join(base_path, 'models')
    submissions_path = os.path.join(base_path, 'submissions')

    # Cria o diretório de submissões se não existir
    os.makedirs(submissions_path, exist_ok=True)

    # Carrega os dados de teste processados
    test_data = pd.read_csv(os.path.join(data_processed_path, 'test_processed.csv'))
    # Carrega o modelo e o scaler
    model, scaler = load_model_and_scaler(models_path)
    # Faz as predições e salva o arquivo de submissão
    make_predictions(test_data, model, scaler, submissions_path)
