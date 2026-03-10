"""
Este script treina uma Rede Adversarial Generativa (GAN) usando PyTorch no conjunto de dados KDEF.
O GAN é composto por um Gerador e um Discriminador. O Gerador gera imagens falsas, enquanto o Discriminador
tenta distinguir entre imagens reais e falsas. O objetivo do GAN é treinar o Gerador para gerar
imagens realistas que podem enganar o Discriminador.

O script usa argumentos de linha de comando para especificar o número de épocas, tamanho do lote, taxa de aprendizado e outros parâmetros.
Ele também usa a biblioteca torchvision para carregar e transformar o conjunto de dados KDEF, e a biblioteca torchsummary para
imprima um resumo dos modelos Gerador e Discriminador.

Durante o treinamento, o script calcula a perda adversária, atualiza os parâmetros Gerador e Discriminador usando
o otimizador Adam e registra os valores de perda usando o TensorBoard. Também salva imagens de amostra geradas pelo Gerador
em intervalos regulares.

"""

import torch
import numpy as np
import argparse
import os
import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.util import img_as_ubyte
from skimage.morphology import square

def selecionar_emocao():
    emocoes = {
        "1": "Felicidade",
        "2": "Medo",
        "3": "Neutro",
        "4": "Raiva",
        "5": "Surpresa",
        "6": "Tristeza"
    }
    
    print("\n=== Seleção de Emoção ===")
    print("Escolha a emoção que deseja treinar:")
    for key, value in emocoes.items():
        print(f"{key} - {value}")
    
    while True:
        escolha = input("\nDigite o número ou nome da emoção: ").strip().lower()
        
        # Verifica se a escolha é um número
        if escolha in emocoes:
            return emocoes[escolha]
        
        # Verifica se a escolha é o nome da emoção
        for key, value in emocoes.items():
            if escolha == value.lower():
                return value
        
        print("Opção inválida! Por favor, tente novamente.")

# Argumentos de linha de comando para número de épocas, tamanho do lote, taxa de aprendizado, etc.
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=8000, help="numero de epocas de treinamento")
parser.add_argument("--batch_size", type=int, default=64, help="tamanho do lote de treinamento")
parser.add_argument("--lr", type=float, default=5e-5, help="adam: taxa de aprendizado")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decaimento do primeiro momento do gradiente")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decaimento do segundo momento do gradiente")
parser.add_argument("--latent_dim", type=int, default=2048, help="dimensão do espaço latente (entrada do gerador)")
parser.add_argument("--img_size", type=int, default=128, help="tamanho da imagem")
parser.add_argument("--channels", type=int, default=1, help="canais da imagem")
parser.add_argument("--output_log_dir", type=str, default="diff-run/py-gan", help="diretório de saída para logs do TensorBoard")
parser.add_argument("--output_img_dir", type=str, default="diff-run/images", help="diretório de saída para imagens geradas")
args = parser.parse_args()

# Selecionar emoção
emocao_escolhida = selecionar_emocao()
print(f"\nTreinando para a emoção: {emocao_escolhida}")

# Definir o caminho dos dados baseado na emoção escolhida
args.data_path = f"./data/Personalizado/{emocao_escolhida}"

# Definir sementes aleatórias para reprodutibilidade
torch.manual_seed(1)

# Crie diretórios para salvar resultados
os.makedirs(args.output_log_dir, exist_ok=True)
os.makedirs(args.output_img_dir, exist_ok=True)

# Criar gravador TensorBoard
writer = SummaryWriter(args.output_log_dir)

# Verifique se CUDA está disponível e configure o dispositivo de acordo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir transformações para os dados de treinamento
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.0], [1.0])
])

# Definir dataset personalizado para KDEF
class KDEFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.JPG"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # No labels needed for GANs

# Carregue o conjunto de dados KDEF e crie um DataLoader para treinamento
train_dataset = KDEFDataset(root_dir=args.data_path, transform=train_transform)
print(f"Number of samples in train_dataset: {len(train_dataset)}")
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

# Defina a forma e a dimensão da imagem
image_shape = (args.channels, args.img_size, args.img_size)
image_dim = int(np.prod(image_shape))

# Defina o modelo do gerador
class Gerador(nn.Module):
    def __init__(self):
        super(Gerador, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, image_dim),
            nn.Tanh()
        )
    
    def forward(self, noise_vector): 
        image = self.model(noise_vector)
        image = image.view(image.size(0), *image_shape)
        return image

# Defina o modelo discriminador
class Discriminador(nn.Module):
    def __init__(self):
        super(Discriminador, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        image_flattened = image.view(image.size(0), -1)
        result = self.model(image_flattened)
        return result

# Crie instâncias dos modelos gerador e discriminador
Gerador = Gerador().to(device)
Discriminador = Discriminador().to(device)

# Imprimir resumo dos modelos gerador e discriminador
summary(Gerador, (args.latent_dim,))
summary(Discriminador, (args.channels, args.img_size, args.img_size))

# Defina a função de perda adversarial
adversarial_loss = nn.BCELoss()

# Definir os otimizadores para os modelos gerador e discriminador
G_optimizer = optim.Adam(Gerador.parameters(), lr=args.lr, betas=(args.b1, args.b2))
D_optimizer = optim.Adam(Discriminador.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Loop de treinamento
num_epochs = 8000
D_loss_plot, G_loss_plot = [], []
for epoch in range(1, args.n_epochs+1): 
    D_loss_list, G_loss_list = [], []
   
    for index, (real_images, _) in enumerate(train_loader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        # Label smoothing
        real_target = Variable(torch.full((real_images.size(0), 1), 0.9, device=device))
        fake_target = Variable(torch.full((real_images.size(0), 1), 0.1, device=device))
        D_real_loss = adversarial_loss(Discriminador(real_images), real_target)
        noise_vector = Variable(torch.randn(real_images.size(0), args.latent_dim).to(device))
        noise_vector = noise_vector.to(device)
        generated_image = Gerador(noise_vector)
        D_fake_loss = adversarial_loss(Discriminador(generated_image), fake_target)
        D_total_loss = D_real_loss + D_fake_loss
        D_loss_list.append(D_total_loss)
        D_total_loss.backward()
        D_optimizer.step()
        G_optimizer.zero_grad()
        generated_image = Gerador(noise_vector)
        G_loss = adversarial_loss(Discriminador(generated_image), real_target)
        G_loss_list.append(G_loss)
        G_loss.backward()
        G_optimizer.step()
        d = generated_image.data
        writer.add_scalar('Perda BCE Discriminador',
                            D_total_loss,
                            epoch * len(train_loader) + index)
        writer.add_scalar('Perda BCE Gerador',
                            G_loss,
                            epoch * len(train_loader) + index)

    # Calcular médias das perdas
    avg_D_loss = torch.mean(torch.FloatTensor(D_loss_list))
    avg_G_loss = torch.mean(torch.FloatTensor(G_loss_list))
    
    print('Epoca: [%d/%d]: Perda_D: %.3f, Perda_G: %.3f' % (
            (epoch), num_epochs, avg_D_loss, avg_G_loss))
    
    D_loss_plot.append(avg_D_loss)
    G_loss_plot.append(avg_G_loss)
    
    # Salvar imagens a cada 50 épocas
    if epoch % 50 == 0:
        # Converter para numpy, aplicar filtro da mediana e voltar para tensor
        imgs = generated_image.data[:90].cpu()
        imgs_np = imgs.numpy()
        imgs_np_filtered = []
        for img in imgs_np:
            img_ = img[0]  # Pega o canal único
            img_ = (img_ - img_.min()) / (img_.max() - img_.min() + 1e-8)  # Normaliza para 0-1
            img_ubyte = img_as_ubyte(img_)
            img_med = median(img_ubyte, square(3))
            img_med = img_med.astype(np.float32) / 255.0  # Volta para float
            imgs_np_filtered.append(img_med)
        imgs_np_filtered = np.stack(imgs_np_filtered)
        imgs_tensor_filtered = torch.tensor(imgs_np_filtered).unsqueeze(1)  # (N, 1, 64, 64)
        save_image(imgs_tensor_filtered, os.path.join(args.output_img_dir, f'sample_{epoch}.png'), nrow=10, normalize=True)

# Após o loop de treinamento
plt.figure(figsize=(10,5))
plt.plot(D_loss_plot, label='Perda Discriminador')
plt.plot(G_loss_plot, label='Perda Gerador')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Perda do Discriminador e Gerador durante o Treinamento')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(args.output_log_dir, 'graficos_perda.png'))
plt.show()
