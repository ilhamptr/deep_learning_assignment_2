# Generative Models pada Fashion-MNIST

Tugas take-home Deep Learning (D6421), Sesi 07 — Generative Models. Proyek ini melatih
dan membandingkan enam model generatif/representasi pada dataset Fashion-MNIST, lalu
mengevaluasinya secara kuantitatif (MSE, PSNR, SSIM, FID), kualitatif (sampling,
interpolasi laten), dan dari sisi efisiensi (waktu latih, latensi inferensi).

## Ringkasan

Semua eksperimen ada di satu notebook: [`deeplearning_assignment2.ipynb`](deeplearning_assignment2.ipynb).

Alur notebook:

1. **MLP Classifier** — membuktikan bahwa bobot dan fitur laten sebuah classifier
   diskriminatif tidak dapat dipakai untuk merekonstruksi atau men-generate citra
   (proyeksi transpos bobot, frozen latent decoder, prior sampling).
2. **Autoencoder** — Vanilla AE dan Sparse AE (penalti sparsity L1 λ‖h‖₁).
3. **VAE** — autoencoder variasional dengan objektif ELBO (BCE + KL).
4. **GAN** — Vanilla GAN (MLP) dan DCGAN (konvolusional penuh).
5. **DDPM** — denoising diffusion probabilistic model (U-Net, 1000 langkah).
6. **Analisis komparatif** — figur interpolasi laten, kurva loss, tabel metrik, dan FID.

## Dataset

Fashion-MNIST (otomatis diunduh via `torchvision.datasets`), 70.000 citra grayscale 28×28:

| Split | Jumlah |
|---|---|
| Train | 56.000 (80%) |
| Val   | 7.000 (10%) |
| Test  | 7.000 (10%) |

Seed tetap `SEED = 42` untuk reproduktibilitas.

## Model yang dibandingkan

| Model | Latent Dim | Objektif |
|---|---|---|
| Vanilla AE | 64 | MSE rekonstruksi |
| Sparse AE | 64 | MSE + λ‖h‖₁ (λ=1e-3) |
| VAE | 64 | ELBO (BCE + KL) |
| Vanilla GAN (MLP) | 100 | Adversarial (JS divergence) |
| DCGAN | 100 | Adversarial (JS divergence) |
| DDPM | — | Prediksi derau (MSE), 1000 langkah |

## Hasil Utama

Diambil dari output notebook (30 epoch, latent dim 64 untuk AE-family):

| Model | Train/Epoch (s) | Latency Batch-64 (ms) | Test MSE | Test PSNR (dB) | Test SSIM | FID |
|---|---|---|---|---|---|---|
| Vanilla AE | 0.87 | 0.85 | 0.0087 | 21.57 | 0.7385 | — |
| Sparse AE | 1.04 | — | 0.0088 | 21.58 | 0.7312 | — |
| VAE | 1.11 | 0.20 | 0.0136 | 19.49 | 0.6403 | 94.8 |
| Vanilla GAN (MLP) | 2.39 | 0.91 | — | — | — | 173.8 |
| DCGAN | 41.33 | 6.42 | — | — | — | 17.3 |
| DDPM | 84.24 | 17126.0 | — | — | — | 30.2 |

Catatan:
- Rekonstruksi terbaik: Vanilla AE ≈ Sparse AE; VAE lebih buruk karena suku KL dan
  sampling stokastik mengorbankan fidelitas per-piksel.
- Kualitas generatif terbaik: **DCGAN** (FID 17.3), lalu DDPM (30.2), VAE (94.8),
  Vanilla GAN (173.8).
- Vanilla AE dan Sparse AE tidak punya FID: ruang laten deterministik memiliki
  *latent voids* sehingga prior sampling gagal.
- Trilemma generatif terlihat dari latensi: generator satu langkah (VAE, GAN, DCGAN)
  jauh lebih cepat daripada DDPM yang butuh 1000 langkah denoising (~17 detik/batch).

## Prasyarat

- Python 3.11
- PyTorch dengan akselerator opsional. Notebook otomatis memilih `mps` (Apple Silicon),
  jika tidak tersedia gunakan CUDA atau CPU.

## Instalasi

```bash
# 1. (opsional) buat virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. instal dependensi
pip install -r requirements.txt
```

Jika `requirements.txt` gagal pada baris `torch`/`torchvision` karena platform berbeda,
instal PyTorch sesuai platform Anda lebih dulu (lihat https://pytorch.org), lalu:

```bash
pip install numpy scipy scikit-learn scikit-image matplotlib pandas pytorch-fid
```

## Menjalankan

```bash
jupyter notebook deeplearning_assignment2.ipynb
```

Jalankan sel dari atas ke bawah. Dataset Fashion-MNIST diunduh otomatis saat pertama
dijalankan. Waktu latih terberat adalah DCGAN (~41 s/epoch) dan DDPM (~84 s/epoch);
pada CPU akan jauh lebih lama.

## Luaran

- Figur: rekonstruksi, sampling generatif, interpolasi laten, kurva loss semua model.
- Metrik FID per model generatif (`pytorch-fid` / InceptionV3).
- `tabel_bagian_iv.csv` — ringkasan metrik kuantitatif seluruh model.

## Struktur

```
.
├── deeplearning_assignment2.ipynb   # seluruh eksperimen
├── requirements.txt
└── README.md
```
