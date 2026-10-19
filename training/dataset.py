# ============================================================
# JEWELLERY DATASET
# ============================================================
#
# Unpaired / self-supervised jewellery dataset.
#
# Important:
#   - Gold and Prototype images are NOT artificially paired.
#   - Two augmented views of the SAME image form the positive
#     pair.
#   - Gold and Prototype remain separate domains.
#
# ============================================================

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class JewelleryDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform,
        image_paths=None,
    ):

        self.root_dir = Path(root_dir)

        self.transform = transform

        # ----------------------------------------------------
        # If paths were supplied by train.py, use them.
        # Otherwise discover all images.
        # ----------------------------------------------------

        if image_paths is not None:

            self.image_paths = image_paths

        else:

            self.image_paths = []

            gold_dir = self.root_dir / "gold"

            prototype_dir = self.root_dir / "prototypes"

            self._add_images(gold_dir, "gold")

            self._add_images(prototype_dir, "prototype")

        self.image_paths.sort(key=lambda item: str(item[0]).lower())

        if not self.image_paths:

            raise RuntimeError("No jewellery images found in:\n" f"{self.root_dir}")

        gold_count = sum(1 for _, domain in self.image_paths if domain == "gold")

        prototype_count = sum(
            1 for _, domain in self.image_paths if domain == "prototype"
        )

        print("=" * 70)
        print("JEWELLERY DATASET")
        print("=" * 70)

        print(f"Gold images:       {gold_count}")

        print(f"Prototype images:  {prototype_count}")

        print(f"Total images:      {len(self.image_paths)}")

    # ========================================================
    # ADD IMAGES
    # ========================================================

    def _add_images(
        self,
        directory,
        domain,
    ):

        if not directory.exists():

            print(f"[WARNING] Directory does not exist:\n" f"{directory}")

            return

        files = [
            path
            for path in directory.rglob("*")
            if (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        ]

        print(f"{domain}: {len(files)} images found")

        for path in files:

            self.image_paths.append((path, domain))

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.image_paths)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path, domain = self.image_paths[index]

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as exc:

            raise RuntimeError(f"Could not load image:\n" f"{image_path}") from exc

        # ----------------------------------------------------
        # Two different augmentations of the SAME image.
        #
        # This is our positive pair.
        # ----------------------------------------------------

        view_1 = self.transform(image)

        view_2 = self.transform(image)

        return {
            "view1": view_1,
            "view2": view_2,
            "path": str(image_path),
            "domain": domain,
        }
