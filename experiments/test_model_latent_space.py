import matplotlib
import torch

matplotlib.use("Agg")

from resonance.evaluation.evaluator import SpectrogramEvaluator
from resonance.evaluation.transition_evaluator import (
    LatentTransitionEvaluator,
    print_chained_transition_results,
)


def main():
    model, _, device, mean, std = SpectrogramEvaluator().load_model_and_dataset(
        checkpoint_name="spectrogram_transition_model_1.pth",
        train_filename="train.npz",
        test_filename="test.npz",
    )

    model.eval()
    evaluator = LatentTransitionEvaluator(
        model=model,
        device=device,
        mean=mean,
        std=std,
        criterion=torch.nn.MSELoss(),
    )
    results = evaluator.evaluate_chained_transitions()

    print_chained_transition_results(results)
    print(f"\nSaved {len(results)} plots to: {results[0]['plot_path'].parent}")


if __name__ == "__main__":
    main()
