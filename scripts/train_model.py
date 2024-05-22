import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Função para treinar e salvar o modelo
def train_and_save_model(train_data, models_path):
    X = train_data.drop(['Survived', 'PassengerId'], axis=1)  # Define as features de treino
    y = train_data['Survived']  # Define a variável alvo
    
    # Divide os dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()  # Inicializa o scaler
    X_train_scaled = scaler.fit_transform(X_train)  # Aplica o scaler nos dados de treino
    X_val_scaled = scaler.transform(X_val)  # Aplica o scaler nos dados de validação
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Inicializa o modelo
    model.fit(X_train_scaled, y_train)  # Treina o modelo
    
    y_val_pred = model.predict(X_val_scaled)  # Faz previsões nos dados de validação
    print(f'Accuracy: {accuracy_score(y_val, y_val_pred)}')  # Exibe a acurácia do modelo
    
    joblib.dump(model, os.path.join(models_path, 'model.pkl'))  # Salva o modelo treinado
    joblib.dump(scaler, os.path.join(models_path, 'scaler.pkl'))  # Salva o scaler

if __name__ == "__main__":
    base_path = 'C:\\Users\\Bruno\\Desktop\\tatanic_a3'  # Define o caminho base
    data_processed_path = os.path.join(base_path, 'data', 'processed')  # Define o caminho dos dados processados
    models_path = os.path.join(base_path, 'models')  # Define o caminho dos modelos

    os.makedirs(models_path, exist_ok=True)  # Cria o diretório de modelos se ele não existir

    train_data = pd.read_csv(os.path.join(data_processed_path, 'train_processed.csv'))  # Carrega os dados de treino processados
    train_and_save_model(train_data, models_path)  # Treina e salva o modelo
