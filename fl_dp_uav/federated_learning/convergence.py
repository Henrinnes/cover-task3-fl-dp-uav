import json
from pathlib import Path
from typing import List, Optional


class ConvergenceTracker:
    """
    Records per-round metrics and detects the convergence round.
    Also tracks the privacy–accuracy tradeoff for analysis.
    """

    def __init__(self, target_accuracy: float = 0.65):
        self.target_accuracy = target_accuracy
        self.rounds: List[int] = []
        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.balanced_accuracies: List[Optional[float]] = []
        self.epsilons: List[float] = []
        self.participant_counts: List[int] = []
        self.convergence_round: Optional[int] = None

    def record(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        epsilon: float,
        num_participants: int,
        balanced_accuracy: Optional[float] = None,
    ):
        self.rounds.append(round_num)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.balanced_accuracies.append(balanced_accuracy)
        self.epsilons.append(epsilon)
        self.participant_counts.append(num_participants)

        if self.convergence_round is None and accuracy >= self.target_accuracy:
            self.convergence_round = round_num

    def summary(self) -> dict:
        return {
            "convergence_round": self.convergence_round,
            "final_accuracy": self.accuracies[-1] if self.accuracies else None,
            "final_loss": self.losses[-1] if self.losses else None,
            "final_epsilon": self.epsilons[-1] if self.epsilons else None,
            "final_balanced_accuracy": (
                self.balanced_accuracies[-1] if self.balanced_accuracies else None
            ),
            "total_rounds": len(self.rounds),
            "avg_participants": (
                sum(self.participant_counts) / len(self.participant_counts)
                if self.participant_counts else 0
            ),
            "target_accuracy": self.target_accuracy,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "rounds": self.rounds,
                    "losses": self.losses,
                    "accuracies": self.accuracies,
                    "balanced_accuracies": self.balanced_accuracies,
                    "epsilons": self.epsilons,
                    "participant_counts": self.participant_counts,
                    "convergence_round": self.convergence_round,
                    "target_accuracy": self.target_accuracy,
                },
                f,
                indent=2,
            )
        print(f"Results saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ConvergenceTracker":
        with open(path) as f:
            data = json.load(f)
        tracker = cls(target_accuracy=data["target_accuracy"])
        tracker.rounds = data["rounds"]
        tracker.losses = data["losses"]
        tracker.accuracies = data["accuracies"]
        tracker.balanced_accuracies = data.get("balanced_accuracies", [None] * len(data["accuracies"]))
        tracker.epsilons = data["epsilons"]
        tracker.participant_counts = data["participant_counts"]
        tracker.convergence_round = data["convergence_round"]
        return tracker
