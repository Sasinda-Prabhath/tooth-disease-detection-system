CREATE TABLE IF NOT EXISTS module2_results (
    id SERIAL PRIMARY KEY,
    patient_ref VARCHAR(64),
    image_key TEXT NOT NULL,
    fdi_number INT,
    angulation VARCHAR(32),
    bone_loss_mm FLOAT,
    severity VARCHAR(16),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
