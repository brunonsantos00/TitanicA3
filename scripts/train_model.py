import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping
import joblib

# Configurar o uso do Keras legado
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Função para treinar e salvar o modelo
def train_and_save_model(train_data, models_path):
    X = train_data.drop(['Survived', 'PassengerId'], axis=1)  # Separa as features (X) removendo 'Survived' e 'PassengerId'
    y = train_data['Survived']  # Define a variável alvo (y)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Divide os dados em treino e validação
    
    scaler = StandardScaler()  # Cria um scaler para normalização dos dados
    X_train_scaled = scaler.fit_transform(X_train)  # Ajusta o scaler e transforma os dados de treino
    X_val_scaled = scaler.transform(X_val)  # Transforma os dados de validação usando o scaler ajustado
    
    # Define a estrutura da rede neural
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Primeira camada oculta com 32 neurônios e ativação ReLU
        Dense(16, activation='relu'),  # Segunda camada oculta com 16 neurônios e ativação ReLU
        Dense(1, activation='sigmoid')  # Camada de saída com 1 neurônio e ativação sigmoide para classificação binária
    ])
    
    # Compila o modelo com otimizador Adam e função de perda binária
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Configura o early stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Treina o modelo
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping])
    
    # Avalia a acurácia do modelo nos dados de validação
    val_accuracy = model.evaluate(X_val_scaled, y_val)[1]
    print(f'Validation Accuracy: {val_accuracy}')
    
    # Salva o modelo treinado e o scaler
    model.save(os.path.join(models_path, 'model.h5'))
    joblib.dump(scaler, os.path.join(models_path, 'scaler.pkl'))

if __name__ == "__main__":
    base_path = r'C:\Users\Bruno\Desktop\tatanic_a3'
    data_processed_path = os.path.join(base_path, 'data', 'processed')
    models_path = os.path.join(base_path, 'models')

    # Cria o diretório para os modelos se não existir
    os.makedirs(models_path, exist_ok=True)

    # Carrega os dados de treino processados
    train_data = pd.read_csv(os.path.join(data_processed_path, 'train_processed.csv'))
    # Treina e salva o modelo
    train_and_save_model(train_data, models_path)
