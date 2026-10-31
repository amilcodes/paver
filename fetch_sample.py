"""Fetch three pinned, openly licensed, privacy-blurred Panoramax images."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone
from PIL import Image

ROOT = Path(__file__).resolve().parent
IDS = {
    'target': '0e5d4027-bf24-48fa-8c5e-8ab13731e095',
    'prev': '9e1423a6-96f2-4a57-9baf-3f0ef11d6949',
    'next': '9c0a39ad-2836-4166-b716-0a87469d361a',
}


def prepare(cache=None, output=ROOT / 'sample', ids=IDS):
    output.mkdir(exist_ok=True)
    records = []
    for name, picture_id in ids.items():
        url = f'https://panoramax.ign.fr/api/pictures/{picture_id}'
        if cache:
            metadata = json.loads((cache / f'{name}.json').read_text())
            data = (cache / f'{name}_hd.jpg').read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                metadata = json.load(response)
            if metadata['properties'].get('license') != 'etalab-2.0':
                raise ValueError('Sample license changed; inspect metadata before fetching')
            with urllib.request.urlopen(metadata['assets']['hd']['href'], timeout=60) as response:
                data = response.read()
        assert metadata['id'] == picture_id
        assert metadata['properties']['license'] == 'etalab-2.0'
        image = Image.open(io.BytesIO(data)).convert('RGB')
        original_size = image.size
        image.resize((6144, 3072), Image.Resampling.LANCZOS).save(
            output / f'{name}.jpg', quality=95, subsampling=0)
        (output / f'{name}.json').write_text(json.dumps(metadata, indent=2) + '\n')
        records.append(dict(name=name, id=picture_id, metadata_url=url,
                            asset_url=metadata['assets']['hd']['href'],
                            source_sha256=hashlib.sha256(data).hexdigest(),
                            sample_sha256=hashlib.sha256((output/f'{name}.jpg').read_bytes()).hexdigest(),
                            original_size=original_size, sample_size=[6144,3072],
                            capture_time=metadata['properties']['datetime'],
                            producer=metadata['properties']['geovisio:producer'],
                            license='etalab-2.0'))
        print('Prepared', name, flush=True)
    (output/'manifest.json').write_text(json.dumps(dict(
        fetched_at_utc=datetime.now(timezone.utc).isoformat(), description='Three consecutive observations from one drive; not a representative benchmark.',
        modifications='Resized from 12288x6144 to 6144x3072; JPEG quality 95. Public blurred assets only.',
        license_url='https://ia.numerique.gouv.fr/licence-ouverte-open-licence/',
        images=records), indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, help='Use previously downloaded name.json and name_hd.jpg files')
    parser.add_argument('--output',type=Path,default=ROOT/'sample')
    parser.add_argument('--ids',type=Path,help='JSON object mapping target, prev, next to picture UUIDs')
    args=parser.parse_args()
    prepare(args.cache,args.output,json.loads(args.ids.read_text()) if args.ids else IDS)
