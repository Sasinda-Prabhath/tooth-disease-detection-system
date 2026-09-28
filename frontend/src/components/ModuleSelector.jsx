import React from 'react';

const MODULES = [
  { id: 'module2', label: 'Module 2 — Impaction & Bone Loss', active: true },
  { id: 'module1', label: 'Module 1 — Caries / Enamel', active: true },
  { id: 'module3', label: 'Module 3 — Decay / Fracture', active: false },
  { id: 'module4', label: 'Module 4 — Gingivitis / Stain', active: false },
];

export default function ModuleSelector({ selected, onSelect }) {
  return (
    <section className="panel module-selector">
      <h3>Disease Module</h3>
      <div className="module-grid">
        {MODULES.map((mod) => (
          <button
            key={mod.id}
            type="button"
            className={`module-chip ${selected === mod.id ? 'selected' : ''} ${mod.active ? '' : 'disabled'}`}
            onClick={() => mod.active && onSelect(mod.id)}
            disabled={!mod.active}
          >
            {mod.label}
            {!mod.active && <span className="soon">Coming soon</span>}
          </button>
        ))}
      </div>
    </section>
  );
}
