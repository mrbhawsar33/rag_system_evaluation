from src.evaluation.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    dataset_path="src/evaluation/question_set.json",
    kb_path="data/processed/vehicle_chunks.json",
    output_path="evaluation_results.csv"
)

runner.run()