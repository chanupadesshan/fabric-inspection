# Data

Nothing in this folder is stored in the repository (see `.gitignore`); only the empty folders are.

## AITEX fabric images (needed for everything)

Put the AITEX Fabric Image Database here, with these three folders:

```
data/raw/
  Defect_images/
  Mask_images/
  NODefect_images/
```

The database was published by AITEX and the Universitat Politecnica de Valencia:
Silvestre-Blanes, Albero-Albero, Miralles, Perez-Llorens, Moreno, "A Public Fabric Database for Defect
Detection Methods and Results", AUTEX Research Journal, December 2019. Follow the download link given in the
paper and respect the dataset's own licence and citation terms.

Then, from the project folder, make the tiles:

```
python src/preprocess.py
```

This fills `data/processed/` (tiles, masks and `tiles.csv`).

## Your own photos (optional)

- `data/real/`: photos to try the models on (`experiments/25_real_photos.ipynb`).
- `data/real_good/`: photos of undamaged fabric of the same type, to calibrate on real fabric.

They stay on your computer and are not uploaded.
