"""S1 step 1: extract the MERFISH cell-by-gene raw counts + cell metadata from the Dryad zip.

Source: Xu RJ, ..., Moffitt JR, Cell Host Microbe 2026 (Dryad doi:10.5061/dryad.p5hqbzm0z),
member data_upload/cell_by_gene/cell_by_gene_with_metadata.h5ad (62.9 GB inflated, 0.99 GB
deflated) of data_upload.zip (26.4 GB). Only the byte range of this member is fetched
(HTTP Range on a presigned Dryad URL, read from a private file, never stored in code); it is
inflated to node-local $TMPDIR, then raw counts are read row-block-wise and written as a
compact sparse (CSR, int32) h5ad with all cell metadata. The inflated file is deleted.

Usage: python s01_extract_merfish.py <url_file> <out_h5ad>
"""
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

HEADER_OFFSET = 161            # from the zip central directory
COMPRESS_SIZE = 990_689_244
FILE_SIZE = 62_895_439_116

url = open(sys.argv[1]).read().strip()
out = Path(sys.argv[2])
tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "s1_extract"
tmp.mkdir(parents=True, exist_ok=True)
deflated = tmp / "member.deflate"
inflated = tmp / "cell_by_gene_with_metadata.h5ad"
t0 = time.time()

# 1) fetch the local file header + compressed data
end = HEADER_OFFSET + 30 + 200 + COMPRESS_SIZE  # header name/extra lengths < 200 bytes
subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "-r", f"{HEADER_OFFSET}-{end}", "-o", str(deflated), url],
               check=True)
with open(deflated, "rb") as f:
    hdr = f.read(30)
    assert hdr[:4] == b"PK\x03\x04", hdr[:4]
    n, e = struct.unpack("<HH", hdr[26:30])
    name = f.read(n).decode()
    f.read(e)
    print("member:", name, "download s:", round(time.time() - t0), flush=True)
    d = zlib.decompressobj(-15)
    written = 0
    with open(inflated, "wb") as g:
        while True:
            chunk = f.read(64 << 20)
            if not chunk:
                break
            buf = d.decompress(chunk)
            g.write(buf)
            written += len(buf)
            if d.eof:
                break
        g.write(d.flush())
print("inflated bytes:", inflated.stat().st_size, "expected:", FILE_SIZE, "s:", round(time.time() - t0), flush=True)
assert inflated.stat().st_size == FILE_SIZE
deflated.unlink()

# 2) inspect structure
with h5py.File(inflated, "r") as h:
    def show(nm, obj):
        if isinstance(obj, h5py.Dataset):
            print(" ", nm, obj.shape, obj.dtype, obj.chunks, obj.compression)
        else:
            print(" ", nm, dict(obj.attrs))
    h.visititems(show)

# 3) metadata via anndata backed mode, raw counts row-block-wise
a = ad.read_h5ad(inflated, backed="r")
obs = a.obs.copy()
var = a.var.copy()
print(a, flush=True)
print(obs.head().to_string(), flush=True)
with h5py.File(inflated, "r") as h:
    node = h["layers/raw_counts"]
    blocks = []
    if isinstance(node, h5py.Dataset):  # dense
        n_obs = node.shape[0]
        step = 50_000
        for i in range(0, n_obs, step):
            x = node[i:i + step]
            assert np.allclose(x, np.round(x)), "raw counts not integer"
            blocks.append(sparse.csr_matrix(x.astype(np.int32)))
            print(f"  rows {i}..{i + len(x)} nnz={blocks[-1].nnz}", flush=True)
        X = sparse.vstack(blocks).tocsr()
    else:  # sparse group
        enc = dict(node.attrs)
        print("sparse raw_counts:", enc, flush=True)
        data = node["data"][:]
        assert np.allclose(data, np.round(data))
        M = (sparse.csr_matrix if "csr" in str(enc.get("encoding-type")) else sparse.csc_matrix)(
            (data.astype(np.int32), node["indices"][:], node["indptr"][:]), shape=tuple(enc["shape"]))
        X = M.tocsr()
X.sort_indices()
print("counts:", X.shape, "nnz:", X.nnz, "total:", int(X.sum()), flush=True)
obs.columns = [str(c) for c in obs.columns]
for c in obs.columns:
    if obs[c].dtype == object:
        obs[c] = obs[c].astype(str)
res = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var.index.astype(str)))
out.parent.mkdir(parents=True, exist_ok=True)
res.write_h5ad(out, compression="gzip")
print("wrote", out, out.stat().st_size / 1e9, "GB; total s:", round(time.time() - t0), flush=True)
inflated.unlink()
