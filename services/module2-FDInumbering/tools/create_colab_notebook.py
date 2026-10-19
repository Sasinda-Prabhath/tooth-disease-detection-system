"""Build the reproducible Colab detector fine-tuning notebook."""
import json
from pathlib import Path
from dental_fdi.paths import PROJECT_ROOT
cells=[]
def markdown(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text):
    cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'outputs':[],'execution_count':None})
markdown('''# FDI detector fine-tuning and automatic angle estimation
This notebook continues training your existing FDI detector rather than restarting from generic weights.
Use a GPU runtime. The image-axis method in the app needs no landmark dataset but produces **provisional, unvalidated estimates**.
Fine-tuning a true root/crown model is a separate optional stage and requires expert landmarks.
Keep patients separated across train/validation/test; review all tooth boxes, especially 17/18, 27/28, 37/38 and 47/48.
More epochs or larger models cannot repair incorrect annotations.''')
code('''!pip install -q ultralytics==8.4.171 torchvision
import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > GPU'
print(torch.cuda.get_device_name(0))''')
markdown('Upload the `FDI_Finetuning_Colab.zip` bundle created by `python -m tools.build_training_bundle`. It contains local X-rays and weights; upload only to a runtime you are authorized to use.')
code('''from google.colab import files
from pathlib import Path
import zipfile, os
uploaded=files.upload()
archive=next(name for name in uploaded if name.endswith('.zip'))
root=Path('/content/fdi_project');root.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        target=(root/member.filename).resolve()
        assert target.is_relative_to(root.resolve()), 'Unsafe archive path'
    z.extractall(root)
os.chdir(root)
print(root)''')
code('''!python -m dental_fdi.training.train --task detect --check-only
# Save in a new run; do not overwrite the starting checkpoint.
!python -m dental_fdi.training.train --task detect --epochs 100 --batch 4 --device 0 --freeze 0 --lr0 0.0001 --name fdi_colab_candidate''')
markdown('Download the actual run selected by the trainer, rather than assuming runs/detect/train. Validate candidate versus baseline before replacing deployed weights.')
code('''import json
summaries=list(Path('runs/detect').glob('*/candidate_summary.json'))
assert summaries, 'Training did not finish; inspect the error above.'
summary=max(summaries,key=lambda p:p.stat().st_mtime)
metadata=json.loads(summary.read_text())
files.download(metadata['best'])
files.download(str(summary))''')
markdown('''## Optional: train a root/crown landmark model
Only run after supplying `pose_dataset/images/{train,val}` and corresponding YOLO pose labels.
Class order: 17, 18, 27, 28, 37, 38, 47, 48. Each row:
`class cx cy width height root_x root_y 2 crown_x crown_y 2` (normalized coordinates).
Annotate all visible target molars in each included image; do not use guessed rectangle endpoints or automatic image-axis estimates as truth.
The trainer checks label structure, not anatomical correctness. Impaction diagnosis needs additional clinical labels and evaluation.''')
code('''from pathlib import Path
if Path('pose_dataset/images/train').is_dir():
    import subprocess
    subprocess.run(['python','-m','dental_fdi.training.train','--task','pose','--check-only'],check=True)
    subprocess.run(['python','-m','dental_fdi.training.train','--task','pose','--weights','yolo26s-pose.pt',
                    '--epochs','150','--batch','4','--device','0','--name','molar_axes'],check=True)
else:
    print('Skipped: no expert root/crown dataset. Use the app image-axis estimation mode instead.')''')
markdown('A dental pose checkpoint must be installed together with `axis_schema.json` from its weights folder. Set `FDI_AXIS_MODEL_PATH` to that checkpoint. Evaluate signed angle error and coverage against an independent annotated test set before relying on its output.')
markdown('''## Optional: train the six-landmark periodontal crop model
Use CLEAN context crops with dentist-annotated mesial/distal CEJ, crest and apex points.
See docs/DATASET_AND_MEASUREMENT_FLOW.md. Tooth boxes or painted bone-loss screenshots are not these labels.
The class is tooth_region with kpt_shape [6,3]. The existing heuristic bone code is not training ground truth.''')
code('''if Path('periodontal_dataset/images/train').is_dir():
    import subprocess
    subprocess.run(['python','-m','dental_fdi.training.train','--task','periodontal','--check-only'],check=True)
    subprocess.run(['python','-m','dental_fdi.training.train','--task','periodontal',
                    '--weights','yolo26s-pose.pt','--epochs','150','--imgsz','512',
                    '--batch','16','--device','0','--name','periodontal_landmarks'],check=True)
else:
    print('Skipped: no expert six-landmark periodontal dataset. Use reviewed manual landmarks in the app.')''')
markdown('Keep the trained periodontal best.pt beside bone_schema.json. Install at the app default path or configure FDI_BONE_MODEL_PATH. Independently evaluate landmark and measurement errors before use.')
notebook={'cells':cells,'metadata':{'language_info':{'name':'python'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':2}
(PROJECT_ROOT / 'notebooks/Dental_FDI_YOLO26_Training.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print('Updated Dental_FDI_YOLO26_Training.ipynb')
