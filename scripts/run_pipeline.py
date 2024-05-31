import subprocess
import os

# Função para executar um script Python e exibir a saída no terminal
def run_script(script_path):
    print(f"Executando {script_path}...")
    try:
        # subprocess.run com check=True para lançar exceção em caso de erro
        subprocess.run(['python', script_path], check=True)
        print(f"Sucesso: {script_path} completou sem erros.\n")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script_path}: {e}\n")

# Função principal para executar os scripts em sequência
def main():
    scripts_dir = 'C:/Users/Bruno/Desktop/tatanic_a3/scripts'
    scripts = ['data_preprocessing.py', 'train_model.py', 'predict.py']
    
    for script in scripts:
        script_path = os.path.join(scripts_dir, script)
        run_script(script_path)  # Executa cada script sequencialmente

if __name__ == "__main__":
    main()
