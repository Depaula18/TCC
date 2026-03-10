import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Caminhos das pastas de logs
pastas = [
    'diff-run\\py-gan-Felicidade',
    'diff-run\\py-gan-Medo',
    'diff-run\\py-gan-Neutro',
    'diff-run\\py-gan-Raiva',
    'diff-run\\py-gan-Surpresa',
    'diff-run\\py-gan-Tristeza'
]

emocoes = ['Felicidade', 'Medo', 'Neutro', 'Raiva', 'Surpresa', 'Tristeza']
cores = ['b', 'g', 'r', 'c', 'm', 'y']

# Gráfico do Discriminador
plt.figure(figsize=(10,6))
for pasta, emocao, cor in zip(pastas, emocoes, cores):
    arquivos = [f for f in os.listdir(pasta) if f.startswith('events.out')]
    if not arquivos:
        continue
    caminho_evento = os.path.join(pasta, arquivos[0])
    ea = event_accumulator.EventAccumulator(caminho_evento)
    ea.Reload()
    if 'Perda BCE Discriminador' in ea.Tags()['scalars']:
        eventos = ea.Scalars('Perda BCE Discriminador')
        valores = [x.value for x in eventos]
        plt.plot(valores, label=emocao, color=cor)
plt.xlabel('Iteração')
plt.ylabel('Perda Discriminador')
plt.title('Perda do Discriminador para cada emoção')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('perda_todas_emocoes_discriminador.png')
plt.show()

# Gráfico do Gerador
plt.figure(figsize=(10,6))
for pasta, emocao, cor in zip(pastas, emocoes, cores):
    arquivos = [f for f in os.listdir(pasta) if f.startswith('events.out')]
    if not arquivos:
        continue
    caminho_evento = os.path.join(pasta, arquivos[0])
    ea = event_accumulator.EventAccumulator(caminho_evento)
    ea.Reload()
    if 'Perda BCE Gerador' in ea.Tags()['scalars']:
        eventos = ea.Scalars('Perda BCE Gerador')
        valores = [x.value for x in eventos]
        plt.plot(valores, label=emocao, color=cor)
plt.xlabel('Iteração')
plt.ylabel('Perda Gerador')
plt.title('Perda do Gerador para cada emoção')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('perda_todas_emocoes_gerador.png')
plt.show()