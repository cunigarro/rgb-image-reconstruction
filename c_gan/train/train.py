import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from generator import Generator
from discrimitator import Discriminator

# Hiperparámetros
latent_dim = 100
num_classes = 10
img_shape = (1, 28, 28)
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 200
sample_interval = 400

# Preparar los datos de MNIST
dataloader = DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Inicializar generador y discriminador
generator = Generator(latent_dim, num_classes, img_shape)
discriminator = Discriminator(num_classes, img_shape)

# Optimizadores
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Pérdida de entropía cruzada binaria
adversarial_loss = torch.nn.BCELoss()

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.size(0)

        # Etiquetas reales y falsas
        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        # Convertir imágenes y etiquetas a tensores
        real_imgs = imgs
        labels = labels

        # Entrenamiento del Generador
        optimizer_G.zero_grad()

        # Generar ruido y etiquetas aleatorias
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))

        # Generar imágenes
        gen_imgs = generator(z, gen_labels)

        # Pérdida del generador
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

        g_loss.backward()
        optimizer_G.step()

        # Entrenamiento del Discriminador
        optimizer_D.zero_grad()

        # Pérdida para imágenes reales
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)

        # Pérdida para imágenes generadas
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)

        # Pérdida total del discriminador
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Imprimir el progreso
        print(
            f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
            f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        )

        # Guardar muestras generadas
        if i % sample_interval == 0:
            gen_imgs = gen_imgs.view(gen_imgs.size(0), *img_shape)
            plt.figure(figsize=(5, 5))
            plt.imshow(gen_imgs[0].squeeze().detach().numpy(), cmap="gray")
            plt.title(f"Generated Image with Label: {gen_labels[0]}")
            plt.show()
