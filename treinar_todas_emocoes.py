import os
import subprocess

base_path = './data/Personalizado'
emocoes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

for emocao in emocoes:
    data_path = os.path.join(base_path, emocao)
    print(f"Treinando para emoção: {emocao}")
    # Define subpastas de saída para cada emoção
    py_gan_dir = f'diff-run/py-gan-{emocao}'
    images_dir = f'diff-run/images-{emocao}'
    os.makedirs(py_gan_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # Chama o script principal passando o caminho da emoção e subpastas de saída
    subprocess.run([
        'python', 'gan_pytorch_fashion_mnist.py',
        '--data_path', data_path,
        '--img_size', '128',
        '--output_log_dir', py_gan_dir,
        '--output_img_dir', images_dir,
    ]) 