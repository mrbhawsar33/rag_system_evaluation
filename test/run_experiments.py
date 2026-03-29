from src.evaluation.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    dataset_path="data/question_set.json",
    kb_path="data/processed/vehicle_chunks.json",
    output_path="data/output/evaluation_results.csv"
)

runner.run()