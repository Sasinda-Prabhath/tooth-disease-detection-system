function PipelineStep({ step, title, description, state }) {
  return (
    <div className="pipeline-step" data-state={state}>
      <span>{step}</span>
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default PipelineStep;
