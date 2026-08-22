import React from 'react';

const MODULES = [
  { id: 'module2', label: 'Impaction & Bone Loss', active: true },
  { id: 'module1', label: 'Caries / Enamel', active: false },
  { id: 'module3', label: 'Decay / Fracture', active: false },
  { id: 'module4', label: 'Gingivitis / Stain', active: false },
];

export default function ModuleSelector({ selected, onSelect }) {
  return (
    <div className="module-selector">
      <div className="module-grid">
        {MODULES.map((mod, index) => (
          <button key={mod.id} type="button" className={`module-chip ${selected === mod.id ? 'selected' : ''} ${mod.active ? '' : 'disabled'}`} onClick={() => mod.active && onSelect(mod.id)} disabled={!mod.active}>
            <small>MODULE {index + 1}</small>{mod.label}{!mod.active && <span className="soon">Coming soon</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
