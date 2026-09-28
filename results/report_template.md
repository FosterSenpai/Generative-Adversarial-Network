## Experiment XX — brief name

- Model name:
- Architecture version:

### Change and reason

### Settings

| Setting                            | Value                                          |
| ---------------------------------- | ---------------------------------------------- |
| Dataset                            | Local cat dataset                              |
| Training images                    | 15,747                                         |
| Input/output                       | 32×32 RGB                                      |
| Preprocessing                      | Bilinear resize, stretch, normalize to [-1, 1] |
| Batch size                         | 256                                            |
| Noise dimensions                   | 100                                            |
| Optimizer                          | Adam                                           |
| Learning rate                      | 0.0001                                         |
| Generator hidden activation        | ReLU                                           |
| Generator output activation        | tanh                                           |
| Discriminator activation           | ReLU                                           |
| Preview noise seed                 | [42, 0]                                        |
| Training random seed               | Not explicitly set                             |
| Completed epochs                   |                                                |
| Generator trainable parameters     |                                                |
| Discriminator trainable parameters |                                                |

### Output comparison

Use the same preview noise and compare matching epochs.

| Epoch | Baseline                                              | This experiment                                              |
| ----- | ----------------------------------------------------- | ------------------------------------------------------------ |
| 50    | ![Baseline at 50](results/cat_gan_v1/epoch_0050.png)  | ![Kernel4 at 50](results/cat_gan_kernel4_v1/epoch_0050.png)  |
| 150   | ![Baseline at 150](results/cat_gan_v1/epoch_0150.png) | ![Kernel4 at 150](results/cat_gan_kernel4_v1/epoch_0150.png) |

### Measurements

| Measurement                            | Baseline | This experiment |
| -------------------------------------- | -------- | --------------- |
| Typical training seconds per epoch     |          |                 |
| Generator loss at comparison epoch     |          |                 |
| Discriminator loss at comparison epoch |          |                 |
