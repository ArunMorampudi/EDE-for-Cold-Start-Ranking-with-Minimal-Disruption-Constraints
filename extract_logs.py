#!/usr/bin/env python
"""Extract gzip compressed Yandex logs."""
import gzip
import os
import sys

data_dir = 'data/yandex'
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks
PROGRESS_INTERVAL = 100 * 1024 * 1024  # Report every 50 MB

for fname in ['train.gz', 'test.gz']:
    gz_path = os.path.join(data_dir, fname)
    out_path = os.path.join(data_dir, fname[:-3] + '.txt')  # Remove .gz and add .txt
    if os.path.exists(gz_path) and not os.path.exists(out_path):
        print(f"Extracting {fname}...")
        
        # Get compressed file size for progress tracking
        compressed_size = os.path.getsize(gz_path)
        compressed_size_mb = compressed_size / (1024 * 1024)
        
        bytes_read = 0
        bytes_written = 0
        last_reported = 0
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    bytes_written += len(chunk)
                    
                    # Report progress every 50 MB extracted
                    if bytes_written - last_reported >= PROGRESS_INTERVAL:
                        # Estimate progress (compressed bytes processed)
                        bytes_read = f_in.fileobj.tell() if hasattr(f_in, 'fileobj') else 0
                        
                        if bytes_read > 0:
                            progress_pct = min(100, (bytes_read / compressed_size) * 100)
                            written_mb = bytes_written / (1024 * 1024)
                            read_mb = bytes_read / (1024 * 1024)
                            print(f"  Progress: {progress_pct:.1f}% ({read_mb:.1f}/{compressed_size_mb:.1f} MB compressed, {written_mb:.1f} MB extracted)")
                        else:
                            written_mb = bytes_written / (1024 * 1024)
                            print(f"  Extracted: {written_mb:.1f} MB")
                        
                        last_reported = bytes_written
        
        print(f"  Created {out_path} ({bytes_written / (1024 * 1024):.1f} MB)")
    elif os.path.exists(out_path):
        print(f"{out_path} already exists")

print("Decompression complete")
