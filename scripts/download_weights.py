"""Download fixed public reference weights and verify SHA-256 before use."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/young-43/2026-computer-ability-challenge-big-data/releases/download/v1.0.0/'


def main():
    config = json.loads((ROOT / 'model/config.json').read_text(encoding='utf-8'))
    for entry in config['weights'].values():
        target = ROOT / 'model' / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']:
                    print('Verified existing', target.name)
                    continue
            raise ValueError('Existing weight has a different hash: ' + target.name)
        temporary = target.with_suffix('.download')
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(BASE + target.name, timeout=120) as response, temporary.open('xb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError('Downloaded weight failed SHA-256: ' + target.name)
            temporary.rename(target)
            print('Downloaded and verified', target.name)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == '__main__':
    main()
