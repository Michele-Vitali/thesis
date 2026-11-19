"""
Build a contact sheet from a recorded HDF5 to check the camera framing.

Rows = episodes, columns = frames evenly spaced over the episode (home -> approach -> grasp -> lift
-> transport -> release). The red box is the 90% center crop the model actually sees at eval time.

Usage:
    python contact_sheet.py <dataset.hdf5> [--demos 6] [--cols 8] [--out contact_sheet.png]
"""
import argparse

import h5py
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--demos", type=int, default=6)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--out", default="contact_sheet.png")
    ap.add_argument("--thumb", type=int, default=224)
    args = ap.parse_args()

    side = np.sqrt(0.9)
    off = (1 - side) / 2 * args.thumb
    box = (off, off, off + side * args.thumb, off + side * args.thumb)

    rows = []
    with h5py.File(args.hdf5, "r") as f:
        names = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[-1]))[: args.demos]
        for name in names:
            imgs = f["data"][name]["obs"]["agentview_rgb"]
            idx = np.linspace(0, len(imgs) - 1, args.cols).astype(int)
            tiles = []
            for i in idx:
                im = Image.fromarray(np.asarray(imgs[i], dtype=np.uint8)).resize(
                    (args.thumb, args.thumb), Image.LANCZOS)
                ImageDraw.Draw(im).rectangle(box, outline=(255, 0, 0), width=1)
                tiles.append(np.asarray(im))
            rows.append(np.concatenate(tiles, axis=1))

    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"Saved {args.out}  ({len(rows)} episodes x {args.cols} frames)")


if __name__ == "__main__":
    main()