"""Package current code/configs and local data without uploading."""
from zipfile import ZipFile, ZIP_DEFLATED
from dental_fdi.paths import PROJECT_ROOT


def main():
    target = PROJECT_ROOT / 'artifacts/FDI_Finetuning_Colab.zip'
    target.parent.mkdir(exist_ok=True)
    files = [PROJECT_ROOT / name for name in ['pyproject.toml', 'requirements.txt', 'README.md', 'app.py']]
    for name in ['dental_fdi', 'configs', 'notebooks', 'dataset', 'pose_dataset', 'periodontal_dataset', 'docs']:
        files.extend(p for p in (PROJECT_ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.tmp'})
    files.append(PROJECT_ROOT / 'runs/detect/train-2/weights/best.pt')
    with ZipFile(target, 'w', ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROJECT_ROOT))
    print(f'Created {target} ({target.stat().st_size / 1024 ** 2:.1f} MB). No upload performed.')


if __name__ == '__main__':
    main()
