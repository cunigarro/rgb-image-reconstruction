# RGB Image Reconstruction For Precision Agriculture

![cgan-architecture](https://github.com/user-attachments/assets/af7492f4-5ce0-487e-ac7c-c0e47ab69e28)

NIR images for Precision Agriculture are generated using cGAN (Conditional Generative Adversarial Network) arquitecture.

Generator: Follows an encoder-decoder or U-Net architecture, where it compresses and then expands the information to generate an image. The use of skip connections in U-Net allows important details to be preserved during the image reconstruction.

Discriminator: Uses a PatchGAN, which evaluates the image in local patches instead of analyzing the entire image, focusing on the texture and local structure.
