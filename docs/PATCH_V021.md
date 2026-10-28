# v0.2.1: storage checksum correction

During the v0.2.0 upload, the Hub automatically added a Git LFS rule for the
captured `passages.jsonl` file to `.gitattributes`. The published checksum manifest
still contained the pre-upload hash for that storage configuration file, so the
full verifier would report one mismatch even though the measured data matched.

v0.2.1 records the actual Hub storage configuration and regenerates the manifest.
No numerical data, model pins, measurement methods or experiment code changed.
The v0.2.0 tag is preserved; use v0.2.1 for complete checksum verification.

- Pre-upload `.gitattributes` SHA-256: `9e75dd981de037ec3769f24f790e126bc5a160b6871f510214e68dc70649aeeb`.
- Hub-generated `.gitattributes` SHA-256: `38aacc121500e33fa1e301d2e9d9d09037c44fe7017730daeff8701713f11d61`.
- Original v0.2.0 commit: `9d01bf24ec5ef34e26b0ccefbb63f0c63696fb20`.
