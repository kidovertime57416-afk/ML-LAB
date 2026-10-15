# LAB08 results

## Configuration comparison

| Config | Conv | Dense | Train % | Val % | Test % | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vgg_6conv | 6 | 64 | 91.30 | 89.13 | 89.13 | 21.0 |
| vgg_8conv | 8 | 128 | 50.00 | 50.00 | 50.00 | 23.6 |
| vgg_10conv | 10 | 128 | 50.00 | 50.00 | 50.00 | 28.6 |

## Epoch comparison (vgg_6conv)

| Epoch | Train % | Val % | Test % | Test loss |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 97.52 | 93.48 | 94.57 | 0.2113 |
| 20 | 91.30 | 89.13 | 89.13 | 0.4689 |
| 30 | 95.65 | 91.30 | 91.30 | 0.1984 |
| 50 | 100.00 | 95.65 | 95.65 | 0.1955 |

Selected by validation: vgg_6conv, epoch 50.
Selected model test accuracy: 95.65%.
