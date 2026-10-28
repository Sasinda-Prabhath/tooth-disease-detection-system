import React, { useState, useEffect } from 'react';
import {
  Eye, EyeOff, Upload, Cpu, Search, FileText, ChevronRight,
  Image, Scan, CheckCircle2, Shield, Phone, MapPin,
  Menu, X, ArrowRight
} from 'lucide-react';
import { predictModule } from './api/client';

/* ───────────────────────────────────────────────
   Data
   ─────────────────────────────────────────────── */
const components = [
  {
    number: '01',
    title: 'Dental Caries & Enamel Erosion Detection',
    description: 'Analyses intraoral photographs to support review of visible caries and enamel erosion.',
    type: 'Intraoral image',
    action: 'Caries analysis',
    image: '/images/components/component-01-caries.jpg',
  },
  {
    number: '02',
    title: 'Impacted Third Molar & Alveolar Bone Loss Detection',
    description: 'Analyses panoramic dental X-rays for third-molar position, angulation, and bone-loss measurements.',
    type: 'Dental X-ray',
    action: 'Bone-loss analysis',
    image: '/images/components/component-02-impaction.jpg',
  },
  {
    number: '03',
    title: 'Tooth Decay & Root Fracture Detection',
    description: 'Supports decay review from clinical images and possible root-fracture region detection from X-rays.',
    type: 'Clinical image',
    action: 'Dental X-ray',
    image: '/images/components/component-03-decay-fracture.jpg',
  },
  {
    number: '04',
    title: 'Gingivitis Severity & Tooth Discoloration Classification',
    description: 'Supports image-based assessment of gingival appearance and tooth discoloration categories.',
    type: 'Intraoral image',
    action: 'Gingival analysis',
    image: '/images/components/component-04-gingivitis.jpg',
  },
];

const workflowSteps = [
  { title: 'Upload Image/X-ray', desc: 'Select a supported clinical image or panoramic dental X-ray.', Icon: Upload },
  { title: 'AI Preprocessing', desc: 'The selected component validates and prepares the image.', Icon: Cpu },
  { title: 'Disease Analysis', desc: 'The component applies its configured detection pipeline.', Icon: Search },
  { title: 'Results & Report', desc: 'Review annotated images, measurements, and structured findings.', Icon: FileText },
];

const supportedInputs = [
  { title: 'Intraoral photographs', desc: 'JPEG, PNG or WebP · Clinical image' },
  { title: 'Panoramic dental X-rays', desc: 'DICOM, PNG · Impaction, bone loss and more' },
  { title: 'Bitewing dental X-rays', desc: 'PNG or JPEG · Supported X-ray' },
  { title: 'Periapical dental X-rays', desc: 'PNG or JPEG · Focused tooth and root area' },
];

const platformBenefits = [
  'Specialised workflows for distinct dental imaging tasks',
  'Visual annotations and structured analysis reports',
  'Clear model status and confidence information',
  'Designed as a supporting tool for dental professionals',
];

/* ───────────────────────────────────────────────
   Logo Component
   ─────────────────────────────────────────────── */
function Logo() {
  return (
    <div className="brand" aria-label="SoftDental">
      <img src="/images/logo.png" alt="SoftDental Logo" className="brand-logo-img" />
    </div>
  );
}

/* ───────────────────────────────────────────────
   Header
   ─────────────────────────────────────────────── */
function Header({ page, setPage }) {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const navigate = (nextPage) => {
    setPage(nextPage);
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollTo = (id) => {
    if (page !== 'home') {
      setPage('home');
      setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }), 100);
    } else {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    }
    setMobileOpen(false);
  };

  return (
    <header className={`site-header${scrolled ? ' scrolled' : ''}`}>
      <button className="logo-button" onClick={() => navigate('home')}>
        <Logo />
      </button>

      <button
        className="mobile-menu-toggle"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle menu"
      >
        {mobileOpen ? <X size={22} /> : <Menu size={22} />}
      </button>

      <nav className={`main-nav${mobileOpen ? ' mobile-open' : ''}`} aria-label="Main navigation">
        <button className={page === 'home' ? 'active' : ''} onClick={() => navigate('home')}>
          Home
        </button>
        <button onClick={() => scrollTo('about')}>About</button>
        <button onClick={() => scrollTo('components')}>Components</button>
        <button onClick={() => scrollTo('how-it-works')}>How It Works</button>
        <button onClick={() => scrollTo('contact')}>Contact</button>
      </nav>

      <div className="header-actions">
        <button className="button button-outline" onClick={() => navigate('signin')}>
          Sign In
        </button>
        <button className="button button-primary" onClick={() => navigate('signup')}>
          Sign Up
        </button>
      </div>
    </header>
  );
}

/* ───────────────────────────────────────────────
   Footer
   ─────────────────────────────────────────────── */
function Footer({ setPage }) {
  const scrollTo = (id) => {
    setPage('home');
    setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }), 100);
  };

  return (
    <footer className="site-footer" id="contact">
      <div className="footer-main">
        {/* Brand Column */}
        <div className="footer-brand">
          <Logo />
          <p>
            AI-assisted dental image analysis designed to support research and professional review
            workflows.
          </p>
          <strong>DECISION-SUPPORT PLATFORM · SRI LANKA</strong>
        </div>

        {/* Quick Links */}
        <div>
          <h4>Quick Links</h4>
          <button className="footer-link" onClick={() => { setPage('home'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>
            Home
          </button>
          <button className="footer-link" onClick={() => scrollTo('about')}>
            About the platform
          </button>
          <button className="footer-link" onClick={() => scrollTo('components')}>
            AI components
          </button>
          <button className="footer-link" onClick={() => scrollTo('how-it-works')}>
            How it works
          </button>
        </div>

        {/* Our Components */}
        <div>
          <h4>Our Components</h4>
          {components.map((c) => (
            <div className="footer-component-link" key={c.number}>
              <span className="footer-component-number">{c.number}</span>
              <span>{c.title.split(' &')[0]}</span>
            </div>
          ))}
        </div>

        {/* Contact */}
        <div className="footer-contact">
          <h4>Public Dental Care</h4>
          <strong>National Dental Hospital (Teaching)</strong>
          <p>
            <MapPin size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
            16 Ward Place
            <br />
            Colombo 07, Sri Lanka
          </p>
          <a href="tel:+94112693106" className="footer-phone">
            <Phone size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 5 }} />
            +94 11 269 3106
          </a>
          <p>For hospital information and public dental-care enquiries.</p>
        </div>
      </div>

      <div className="footer-bottom">
        <span>© 2026 Tooth Disease Detection System.</span>
        <span>AI research and professional decision support</span>
      </div>
    </footer>
  );
}

/* ───────────────────────────────────────────────
   Home Page
   ─────────────────────────────────────────────── */
function Home({ setPage }) {
  return (
    <div className="page-enter">
      {/* ── Hero ──────────────────────────────── */}
      <section className="hero">
        <div className="hero-copy animate-in">
          <span className="section-kicker">SMART DENTAL IMAGE ANALYSIS</span>
          <h1>
            AI-Powered
            <br />
            <em>Multi-Disease</em> Dental
            <br />
            Detection and
            <br />
            Prediction System
          </h1>
          <p>
            Analyse intraoral photographs and dental X-rays with tools designed to support
            dental-condition detection, measurement, and professional review.
          </p>
          <div className="hero-actions">
            <button
              className="button button-primary button-large"
              onClick={() => setPage('signup')}
            >
              Get Started <span className="btn-arrow">→</span>
            </button>
            <button
              className="button button-outline button-large"
              onClick={() =>
                document.getElementById('components')?.scrollIntoView({ behavior: 'smooth' })
              }
            >
              Explore Components
            </button>
          </div>
          <div className="hero-highlights">
            <span>
              <Image size={14} className="highlight-icon" /> Image-guided analysis
            </span>
            <span>
              <FileText size={14} className="highlight-icon" /> Structured results
            </span>
            <span>
              <Shield size={14} className="highlight-icon" /> Professional support tool
            </span>
          </div>
        </div>

        <div className="hero-visual animate-in animate-in-delay-2">
          <div className="hero-image">
            <img
              src="https://images.unsplash.com/photo-1629909613654-28e377c37b09?w=800&h=500&fit=crop"
              alt="Dentist reviewing dental scan on monitor"
              loading="eager"
            />
            <div className="hero-image-overlay" />
            <div className="hero-scan-effect" />
          </div>
          <div className="visual-labels">
            <div className="visual-label-card">
              <span className="label-icon blue">
                <Scan size={18} />
              </span>
              <div>
                <small>ILLUSTRATIVE PREVIEW</small>
                <strong>Dental scan review</strong>
              </div>
            </div>
            <div className="visual-label-card">
              <span className="label-icon green">
                <CheckCircle2 size={18} />
              </span>
              <div>
                <small>ANALYSIS WORKFLOW</small>
                <strong>Review-ready output</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Feature Strip ─────────────────────── */}
      <div className="feature-strip">
        {[
          { Icon: Eye, title: 'Visual annotations', desc: 'Review detected regions directly on dental images.' },
          { Icon: FileText, title: 'Structured results', desc: 'Organised findings and component-specific reports.' },
          { Icon: Scan, title: 'Modality checks', desc: 'Input validation for clinical images and X-rays.' },
          { Icon: Shield, title: 'Transparent support', desc: 'Clear model status and professional review context.' },
        ].map(({ Icon, title, desc }) => (
          <div className="feature-item" key={title}>
            <span className="feature-icon">
              <Icon size={16} />
            </span>
            <section>
              <strong>{title}</strong>
              <p>{desc}</p>
            </section>
          </div>
        ))}
      </div>

      {/* ── Components Section ────────────────── */}
      <section className="section-block" id="components">
        <div className="center-heading">
          <span className="section-kicker">EXPLORE THE PLATFORM</span>
          <h2>Our 4 Integrated AI Components</h2>
          <p>
            Purpose-built workflows for different dental images and conditions, with availability
            shown clearly.
          </p>
        </div>
        <div className="component-grid">
          {components.map((c) => (
            <article className="component-card" key={c.number}>
              <div className="component-image">
                <img src={c.image} alt={c.title} loading="lazy" />
                <div className="component-image-overlay" />
                <span className="component-number">{c.number}</span>
              </div>
              <div className="component-content">
                <h3>{c.title}</h3>
                <p>{c.description}</p>
                <div className="card-tags">
                  <span>{c.type}</span>
                  <span>{c.action}</span>
                </div>
                <button className="card-link" onClick={() => c.number === '01' ? setPage('analysis') : setPage('signup')}>
                  Open Component {c.number} <span className="card-link-arrow">→</span>
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* ── Lower Grid ────────────────────────── */}
      <section className="lower-grid" id="how-it-works">
        {/* How It Works */}
        <div className="workflow-card" id="about">
          <span className="section-kicker">SIMPLE WORKFLOW</span>
          <h2>How It Works</h2>
          {workflowSteps.map((step, i) => (
            <div className="workflow-step" key={step.title}>
              <span className="workflow-step-number">0{i + 1}</span>
              <span className="workflow-step-icon">
                <step.Icon size={16} />
              </span>
              <div>
                <strong>{step.title}</strong>
                <p>{step.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Info Card */}
        <div className="info-card">
          <div>
            <span className="section-kicker">COMPATIBLE MEDIA</span>
            <h2>Supported Inputs</h2>
            {supportedInputs.map((item) => (
              <div className="input-row" key={item.title}>
                <span className="input-row-icon">
                  <Image size={14} />
                </span>
                <div>
                  <strong>{item.title}</strong>
                  <br />
                  <small style={{ color: '#a0adb8', fontSize: '9px' }}>{item.desc}</small>
                </div>
              </div>
            ))}
          </div>
          <div>
            <span className="section-kicker">BUILT FOR CLARITY</span>
            <h2>Why Choose Our Platform</h2>
            {platformBenefits.map((item) => (
              <div className="check-row" key={item}>
                <span className="check-icon">✓</span>
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function AnalysisPage({ setPage }) {
  const [selectedDiseases, setSelectedDiseases] = useState({ caries: true, enamel: true });
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFile = (event) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    setFile(nextFile);
    setResults(null);
    setError('');
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(nextFile));
  };

  const analyze = async () => {
    if (!file) {
      setError('Please choose an X-ray or clinical image first.');
      return;
    }
    if (!selectedDiseases.caries && !selectedDiseases.enamel) {
      setError('Select at least one condition to analyze.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      setResults(await predictModule('module1', file));
    } catch (err) {
      setError(err.message || 'The analysis service is unavailable. Start Module 1 on port 8001 and try again.');
    } finally {
      setLoading(false);
    }
  };

  const toggleDisease = (key) => {
    setSelectedDiseases((current) => ({ ...current, [key]: !current[key] }));
  };

  return (
    <main className="analysis-page page-enter">
      <div className="analysis-heading">
        <button className="back-link" onClick={() => setPage('home')}>← Back to components</button>
        <span className="section-kicker">COMPONENT 01 · DENTAL IMAGE ANALYSIS</span>
        <h1>Caries &amp; Enamel Erosion Detection</h1>
        <p>Choose the conditions you want to review, upload an intraoral image or dental X-ray, and run the configured analysis models.</p>
      </div>

      <div className="analysis-layout">
        <section className="analysis-card setup-card">
          <div className="analysis-card-title"><span className="step-badge">01</span><div><h2>Select conditions</h2><p>Choose one or both predictions.</p></div></div>
          <div className="disease-options">
            <button className={`disease-option ${selectedDiseases.caries ? 'selected' : ''}`} onClick={() => toggleDisease('caries')}>
              <span className="option-check">{selectedDiseases.caries ? '✓' : ''}</span><span><strong>Dental caries</strong><small>Detect and grade visible caries.</small></span>
            </button>
            <button className={`disease-option ${selectedDiseases.enamel ? 'selected' : ''}`} onClick={() => toggleDisease('enamel')}>
              <span className="option-check">{selectedDiseases.enamel ? '✓' : ''}</span><span><strong>Enamel erosion</strong><small>Identify erosion and affected surface.</small></span>
            </button>
          </div>

          <div className="analysis-card-title upload-title"><span className="step-badge">02</span><div><h2>Upload image</h2><p>PNG, JPG, JPEG or supported clinical image.</p></div></div>
          <label className="upload-dropzone">
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={handleFile} />
            <Upload size={25} />
            <strong>{file ? file.name : 'Choose an X-ray or image'}</strong>
            <small>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'Drag and drop here, or browse from your device'}</small>
            <span className="browse-button">Browse files</span>
          </label>
          {previewUrl && <ImagePreview previewUrl={previewUrl} results={results} selectedDiseases={selectedDiseases} />}
          {error && <div className="analysis-error">{error}</div>}
          <button className="button button-primary analyze-button" onClick={analyze} disabled={loading}>
            {loading ? 'Analyzing image…' : 'Analyze image'} <ArrowRight size={15} />
          </button>
        </section>

        <section className="analysis-card results-card">
          <div className="analysis-card-title"><span className="step-badge">03</span><div><h2>Prediction results</h2><p>Review the model output for your selected image.</p></div></div>
          {!results && !loading && <div className="empty-results"><FileText size={32} /><strong>Your results will appear here</strong><span>Upload an image and select Analyze image to begin.</span></div>}
          {loading && <div className="empty-results"><span className="loading-ring" /><strong>Analyzing your image</strong><span>The configured models are processing the selected file.</span></div>}
          {results && <Results results={results} selectedDiseases={selectedDiseases} />}
        </section>
      </div>
      <div className="analysis-note"><Shield size={16} /><span>This tool provides decision support for research and professional review. Predictions should be reviewed by a qualified dental professional.</span></div>
    </main>
  );
}

function ImagePreview({ previewUrl, results, selectedDiseases }) {
  const width = results?.image_width || 1;
  const height = results?.image_height || 1;
  const annotations = (results?.annotations || []).filter((annotation) => {
    const source = `${annotation.source || ''} ${annotation.label || ''}`.toLowerCase();
    return (selectedDiseases.caries && source.includes('caries'))
      || (selectedDiseases.enamel && source.includes('enamel'));
  });
  return <div className="image-preview annotated-preview">
    <div className="annotation-stage">
      <img src={previewUrl} alt="Selected dental image preview" />
      {annotations.map((annotation, index) => {
        const left = Math.max(0, (annotation.x1 / width) * 100);
        const top = Math.max(0, (annotation.y1 / height) * 100);
        const boxWidth = Math.max(0, ((annotation.x2 - annotation.x1) / width) * 100);
        const boxHeight = Math.max(0, ((annotation.y2 - annotation.y1) / height) * 100);
        return <span
          className="annotation-box"
          key={`${annotation.source || annotation.label}-${index}`}
          style={{ left: `${left}%`, top: `${top}%`, width: `${boxWidth}%`, height: `${boxHeight}%` }}
          title={`${annotation.label || 'Detection'} · ${(annotation.confidence * 100).toFixed(1)}%`}
        >
          <b>{annotation.label || 'Detection'}</b>
        </span>;
      })}
    </div>
    {annotations.length > 0 && <small className="annotation-caption">Red boxes show detected regions from the selected disease models.</small>}
  </div>;
}

function Results({ results, selectedDiseases }) {
  const selectedTeeth = results.teeth || [];
  const cariesSeverity = (confidence) => {
    if (typeof confidence !== 'number') return 'Unavailable';
    const percentage = confidence * 100;
    if (percentage < 20) return 'None';
    if (percentage < 50) return 'Low';
    if (percentage <= 80) return 'Moderate';
    return 'Advanced';
  };
  const average = (key) => {
    const values = selectedTeeth.map((tooth) => key(tooth)).filter((value) => typeof value === 'number');
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const percent = (value) => value === null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;
  const cariesConfidence = average((tooth) => tooth.caries?.confidence);
  const enamelConfidence = typeof results.enamel_erosion?.confidence === 'number'
    ? results.enamel_erosion.confidence
    : average((tooth) => tooth.enamel_erosion?.confidence);
  const primaryEnamelModel = Object.entries(results.model_results || {})
    .filter(([name]) => name.toLowerCase().includes('enamel'))
    .reduce((best, current) => {
      if (!best || (current[1].confidence || 0) > (best[1].confidence || 0)) return current;
      return best;
    }, null);
  const isSelectedModel = (name) => {
    const normalized = name.toLowerCase();
    return (selectedDiseases.caries && normalized.includes('caries'))
      || (selectedDiseases.enamel && normalized.includes('enamel'));
  };
  const selectedModelResults = Object.entries(results.model_results || {})
    .filter(([name]) => isSelectedModel(name));
  const selectedModelStatus = Object.entries(results.model_status || {})
    .filter(([name]) => {
      const normalized = name.toLowerCase();
      return (selectedDiseases.caries && normalized.includes('caries'))
        || (selectedDiseases.enamel && normalized.includes('enamel'));
    });
  return <div className="results-content">
    <div className="result-status"><CheckCircle2 size={16} /> Analysis complete <span>{selectedTeeth.length} region{selectedTeeth.length === 1 ? '' : 's'} reviewed</span></div>
    <div className="result-metrics">
      {selectedDiseases.caries && <div className="result-metric"><small>Dental caries confidence</small><strong>{percent(cariesConfidence)}</strong><span>Classification model</span></div>}
      {selectedDiseases.enamel && <div className="result-metric enamel"><small>Enamel erosion confidence</small><strong>{percent(enamelConfidence)}</strong><span>Erosion detection model</span></div>}
    </div>
    <div className="model-results-grid">
      {selectedModelResults.map(([name, output]) => (
        <article className="model-result-card" key={name}>
          <div><strong>{name}</strong><small>{output.type}</small></div>
          <b>{typeof output.confidence === 'number' ? percent(output.confidence) : percent(output.average_confidence)}</b>
          <span>
            {output.label ? `${output.label} · ` : ''}
            {typeof output.detections === 'number' ? `${output.detections} detections` : `${output.predictions || 0} predictions`}
          </span>
        </article>
      ))}
    </div>
    {selectedDiseases.enamel && results.enamel_erosion && <div className="image-level-result"><div><strong>{primaryEnamelModel?.[0] || 'Enamel model'} result</strong><small>Strongest available enamel prediction</small></div><span><b>{results.enamel_erosion.label}</b> · {percent(results.enamel_erosion.confidence)}<small>{results.enamel_erosion.affected_surface_pct.toFixed(1)}% affected surface</small></span></div>}
    {selectedDiseases.caries && (selectedTeeth.length === 0 ? <div className="no-detections">No caries regions were detected in this image.</div> : <div className="prediction-list">{selectedTeeth.map((tooth, index) => <article key={`${tooth.fdi_number}-${index}`}><div><strong>{tooth.fdi_number ? `FDI ${tooth.fdi_number}` : `Detection ${index + 1}`}</strong><small>Caries severity grading</small></div><div className="prediction-values">{tooth.caries && <span><b>{cariesSeverity(tooth.caries.confidence)}</b> · {percent(tooth.caries.confidence)}</span>}</div></article>)}</div>)}
    <div className="model-status-row">{selectedModelStatus.map(([name, loaded]) => <span key={name} className={loaded ? 'loaded' : ''}><i />{name.replaceAll('_', ' ')} · {loaded ? 'ML model' : 'fallback'}</span>)}</div>
  </div>;
}

/* ───────────────────────────────────────────────
   Account Page (Sign In / Sign Up)
   ─────────────────────────────────────────────── */
function AccountPage({ type, setPage }) {
  const isSignUp = type === 'signup';
  const [showPass, setShowPass] = useState(false);
  const [showConfirmPass, setShowConfirmPass] = useState(false);

  const benefits = [
    ...(isSignUp
      ? [
          { num: '01', title: 'Your Account', desc: 'Access your dental analysis workspace.' },
          { num: '02', title: 'AI-Assisted Analysis', desc: 'Explore supported dental image analysis tools.' },
          { num: '03', title: 'Detailed Reports', desc: 'Review available findings and analysis results.' },
          { num: '04', title: 'Simple Workflow', desc: 'Upload images and follow a clear analysis process.' },
        ]
      : [
          { num: '01', title: 'AI-Assisted Analysis', desc: 'Explore supported dental image analysis tools.' },
          { num: '02', title: 'Detailed Reports', desc: 'Review available findings and analysis results.' },
          { num: '03', title: 'Simple Workflow', desc: 'Upload images and follow a clear analysis process.' },
        ]),
  ];

  return (
    <main className="account-page page-enter">
      {/* Left Intro */}
      <div className="account-intro">
        {isSignUp && (
          <div className="signup-intro-graphic">
            <span className="signup-graphic-tooth">⚕</span>
            <div className="signup-graphic-nodes">
              <div className="graphic-node" />
              <div className="graphic-node" />
              <div className="graphic-node" />
              <div className="graphic-node" />
              <div className="graphic-node" />
              <div className="graphic-node" />
            </div>
          </div>
        )}

        <span className="section-kicker">
          {isSignUp ? 'CREATE YOUR ACCOUNT' : 'DENTALAI ACCOUNT'}
        </span>
        <h1>
          {isSignUp ? (
            <>
              Join <em>DentalAI</em> Today
            </>
          ) : (
            <>
              Welcome back to <em>DentalAI</em>
            </>
          )}
        </h1>
        <p>
          {isSignUp
            ? 'Create your account to access AI-assisted dental image analysis and view your reports.'
            : 'Sign in to access your dental image analysis workspace and review your reports.'}
        </p>

        <div className="account-benefits">
          {benefits.map((b) => (
            <div className="account-benefit-item" key={b.num}>
              <span className="benefit-number">{b.num}</span>
              <div>
                <strong>{b.title}</strong>
                <p>{b.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right Form */}
      <form className="account-form" onSubmit={(e) => e.preventDefault()}>
        <span className="section-kicker">DENTALAI ACCOUNT</span>
        <h2>{isSignUp ? 'Sign Up' : 'Sign In'}</h2>
        <p>{isSignUp ? 'Create your account to get started.' : 'Access your account to continue.'}</p>

        {isSignUp && (
          <label>
            Full Name
            <input placeholder="Enter your full name" autoComplete="name" />
          </label>
        )}

        <label>
          Email Address
          <input type="email" placeholder="name@example.com" autoComplete="email" />
        </label>

        {isSignUp && (
          <div className="form-row">
            <label>
              Username
              <input placeholder="Choose a username" autoComplete="username" />
            </label>
            <label>
              Phone Number
              <small style={{ color: '#a0adb8', fontWeight: 400 }}>(Optional)</small>
              <input placeholder="+94 7X XXX XXXX" type="tel" autoComplete="tel" />
            </label>
          </div>
        )}

        <label className="password-field">
          Password
          <input
            type={showPass ? 'text' : 'password'}
            placeholder={isSignUp ? 'Create a secure password' : 'Enter your password'}
            autoComplete={isSignUp ? 'new-password' : 'current-password'}
          />
          <button
            type="button"
            className="password-toggle"
            onClick={() => setShowPass(!showPass)}
            aria-label={showPass ? 'Hide password' : 'Show password'}
          >
            {showPass ? <EyeOff size={15} /> : <Eye size={15} />}
            {' '}{showPass ? 'Hide' : 'Show'}
          </button>
        </label>

        {isSignUp && (
          <>
            <div className="password-criteria">
              <span><span className="criteria-dot" /> 8+ characters</span>
              <span><span className="criteria-dot" /> Uppercase letter</span>
              <span><span className="criteria-dot" /> Lowercase letter</span>
              <span><span className="criteria-dot" /> Number</span>
            </div>

            <label className="password-field">
              Confirm Password
              <input
                type={showConfirmPass ? 'text' : 'password'}
                placeholder="Re-enter your password"
                autoComplete="new-password"
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowConfirmPass(!showConfirmPass)}
                aria-label={showConfirmPass ? 'Hide password' : 'Show password'}
              >
                {showConfirmPass ? <EyeOff size={15} /> : <Eye size={15} />}
                {' '}{showConfirmPass ? 'Hide' : 'Show'}
              </button>
            </label>

            <label className="checkbox-label">
              <input type="checkbox" />
              I agree to the Terms of Service and Privacy Policy.
            </label>
          </>
        )}

        <button className="button button-primary button-full" style={{ marginTop: 18 }}>
          {isSignUp ? 'Create Account' : 'Sign In'}
        </button>

        <p className="form-switch">
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button type="button" onClick={() => setPage(isSignUp ? 'signin' : 'signup')}>
            {isSignUp ? 'Sign in' : 'Sign up'}
          </button>
        </p>
        <small>Your password is securely hashed before it is stored.</small>
      </form>
    </main>
  );
}

/* ───────────────────────────────────────────────
   App Root
   ─────────────────────────────────────────────── */
export default function App() {
  const [page, setPage] = useState('home');

  // Reset scroll on page change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [page]);

  const renderPage = () => {
    switch (page) {
      case 'signin':
        return <AccountPage type="signin" setPage={setPage} />;
      case 'signup':
        return <AccountPage type="signup" setPage={setPage} />;
      case 'analysis':
        return <AnalysisPage setPage={setPage} />;
      default:
        return <Home setPage={setPage} />;
    }
  };

  return (
    <div className="site-shell">
      <Header page={page} setPage={setPage} />
      <div className="page-content">{renderPage()}</div>
      <Footer setPage={setPage} />
    </div>
  );
}
