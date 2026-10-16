function ProcessingStepper({ activeIndex = 0, complete = false }) {
  const steps = ['Image Received', 'Preprocessing', 'AI Analysis', 'Results Generation', 'Report Preparation'];

  return (
    <div className="processing-stepper">
      {steps.map((step, index) => {
        const state = complete || index < activeIndex ? 'complete' : index === activeIndex ? 'active' : 'idle';
        return (
          <div key={step} className="stepper-item" data-state={state}>
            <span>{index + 1}</span>
            <strong>{step}</strong>
          </div>
        );
      })}
    </div>
  );
}

export default ProcessingStepper;
