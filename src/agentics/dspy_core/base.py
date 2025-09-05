from abc import abstractmethod, ABCMeta
import dspy
import json
import platform
import datetime
from pathlib import Path
from typing import Optional, Literal
from dspy.teleprompt.teleprompt import Teleprompter


class ProgramMeta(type(dspy.Module), ABCMeta):
    pass


class Program(dspy.Module, metaclass=ProgramMeta):
    def __init__(self, lm: dspy.LM = None, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.lm: dspy.LM = lm or dspy.settings.lm
        self.optimizer: Teleprompter = None
        self.optimized_program = None
        self.score = None

    ##### Required Methods #####
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward method for prediction - must be implemented by subclass"""
        pass

    ##### Required for optimization #####
    @property
    def dataset(self):
        """Full dataset, list[dspy.Example] - optional, used for training"""
        return []

    @property
    def trainset(self):
        """Training set, list[dspy.Example] - optional, used for training"""
        return []

    @property
    def valset(self):
        """Validation set, list[dspy.Example] - optional, used for validation"""
        return []

    @staticmethod
    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        """Evaluation metric for this program - optional, used for evaluation"""
        return 0.0

    ##### Optimizer Metadata #####
    @property
    def optimizer_name(self) -> str:
        """Get the optimizer name from metadata"""
        if self.optimizer is None:
            raise ValueError("No optimizer found. This program was not optimized.")
        return self.optimizer.__class__.__name__

    def _get_optimizer_params(self) -> dict:
        if self.optimizer is None:
            raise ValueError("No optimizer found. This program was not optimized.")

        params = self.optimizer.get_params()
        json_serializable_params = {}

        for key, value in params.items():
            try:
                json.dumps(value)  # Test if it's JSON serializable
                json_serializable_params[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue

        return json_serializable_params

    ##### Evaluate #####
    def evaluate(self, num_threads: int = 10):
        evaluator = dspy.Evaluate(
            devset=self.valset,
            num_threads=num_threads,
            display_progress=True,
            metric=self.metric,
        )
        result = evaluator(self)
        self.score = result.score
        return result

    ##### Ensure Requirements #####

    def ensure_metric(self):
        """Check if the metric is properly implemented."""
        errors = []

        if not callable(self.metric):
            errors.append(
                "❌ Metric is not implemented. You must define a 'metric' method that takes "
                "(example: dspy.Example, prediction: dspy.Prediction, trace=None) and returns a float score."
            )
        elif self.metric == Program.metric:
            # Check if it's still the default implementation
            errors.append(
                "❌ Metric is using default implementation. You must override the 'metric' method "
                "with your own evaluation logic that returns a meaningful score."
            )

        if errors:
            error_message = "\n".join(errors)
            raise ValueError(
                f"❌ Metric requirement not met:\n{error_message}\n\n"
                "To fix this, implement the metric in your subclass:\n\n"
                "Example:\n"
                "@staticmethod\n"
                "def metric(example, prediction, trace=None):\n"
                "    return 1.0 if prediction.answer == example.answer else 0.0"
            )

    def ensure_trainset(self):
        """Check if the trainset is properly implemented."""
        errors = []
        warnings = []

        if not isinstance(self.trainset, list):
            errors.append(
                "❌ trainset must be a list of dspy.Example objects."
            )
        elif len(self.trainset) == 0:
            warnings.append(
                "⚠️  trainset is empty. Optimization will not be effective without training examples."
            )
        else:
            # Check if all items are dspy.Example
            invalid_items = [i for i, item in enumerate(self.trainset) if not isinstance(item, dspy.Example)]
            if invalid_items:
                errors.append(
                    f"❌ trainset contains invalid items at indices {invalid_items}. "
                    "All items must be dspy.Example objects."
                )

        # Show warnings
        for warning in warnings:
            print(warning)

        if errors:
            error_message = "\n".join(errors)
            raise ValueError(
                f"❌ Trainset requirement not met:\n{error_message}\n\n"
                "To fix this, implement the trainset in your subclass:\n\n"
                "Example:\n"
                "@property\n"
                "def trainset(self):\n"
                "    return [dspy.Example(question='...', answer='...')]"
            )

    def ensure_valset(self):
        """Check if the valset is properly implemented."""
        errors = []
        warnings = []

        if not isinstance(self.valset, list):
            errors.append(
                "❌ valset must be a list of dspy.Example objects."
            )
        elif len(self.valset) == 0:
            warnings.append(
                "⚠️  valset is empty. You won't be able to evaluate optimization performance."
            )
        else:
            # Check if all items are dspy.Example
            invalid_items = [i for i, item in enumerate(self.valset) if not isinstance(item, dspy.Example)]
            if invalid_items:
                errors.append(
                    f"❌ valset contains invalid items at indices {invalid_items}. "
                    "All items must be dspy.Example objects."
                )

        # Show warnings
        for warning in warnings:
            print(warning)

        if errors:
            error_message = "\n".join(errors)
            raise ValueError(
                f"❌ Valset requirement not met:\n{error_message}\n\n"
                "To fix this, implement the valset in your subclass:\n\n"
                "Example:\n"
                "@property\n"
                "def valset(self):\n"
                "    return [dspy.Example(question='...', answer='...')]"
            )


    ##### Save and Load #####

    def get_metadata(self) -> dict:

        metadata = {
            "python_version": platform.python_version(),
            "dspy_version": dspy.__version__,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        }

        if not self.optimizer:
            return metadata

        if self.score is None:
            print("⏳ Evaluating score...")
            self.evaluate()

        optimization_metadata = {
            "score": self.score,
            "optimizer_name": self.optimizer_name,
            "optimizer_params": self._get_optimizer_params(),
        }

        metadata.update(optimization_metadata)
        return metadata

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save this program.
        Overrides dspy's save method to add automatic versioning when no path is provided.

        Args:
            path: Explicit path to save to. If None, uses auto-versioning with optimizer metadata.

        Returns:
            Path: Full path where the model was saved
        """
        if path is not None:
            path = Path(path)
            super().save(path)
            print(f"💾 Saved model: {path}")
            return path

        if not self.optimizer:
            raise ValueError("No optimizer found. This program was not optimized.")

        optimizer_name = self.optimizer.__class__.__name__

        script_dir = Path(__file__).parent
        base_path = script_dir / "optim"

        # Create directory structure: base_path / model_name / optimizer_name
        model_name = str(self.lm.model).split("/")[-1]
        folder = base_path / model_name / optimizer_name
        folder.mkdir(parents=True, exist_ok=True)

        # Find next available version number (up to 999)
        for version in range(1, 1000):
            auto_path = folder / f"v{version}.json"
            metadata_path = folder / f"v{version}_metadata.json"
            if not auto_path.exists() and not metadata_path.exists():
                break
        else:
            raise RuntimeError("No available version slot found in 1000 attempts.")

        # Call dspy's original save method
        super().save(auto_path)
        metadata = self.get_metadata()
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f"💾 Saved optimized model to: {auto_path}")

        return auto_path

    def load(
        self,
        path: Path | str | None = None,
        version: int | None = None,
        optimizer: Teleprompter | None = None,
    ):
        program_score = None
        if path is not None:
            # If path is provided, just load from that path
            path = Path(path)
            super().load(path)
            # Try to load metadata if it exists
            metadata_path = path.parent / f"{path.stem}_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    self.score = metadata.get("score")
                except (json.JSONDecodeError, KeyError):
                    self.score = None
            else:
                self.score = None
        elif optimizer is not None:
            # Extract optimizer name
            optimizer_name = optimizer.__class__.__name__

            model_name = str(self.lm.model).split("/")[-1]

            # Construct base path
            script_dir = Path(__file__).parent
            base_path = script_dir / "optim"
            folder = base_path / model_name / optimizer_name

            if not folder.exists():
                raise FileNotFoundError(f"Optimizer folder not found: {folder}")

            if version is not None:
                # Use specific version
                load_path = folder / f"v{version}.json"
                metadata_path = folder / f"v{version}_metadata.json"
                if not load_path.exists():
                    raise FileNotFoundError(f"Version {version} not found: {load_path}")
            else:
                # Find latest version by checking in reverse order
                load_path = None
                metadata_path = None
                for version_num in range(999, 0, -1):  # 999 down to 1
                    candidate_path = folder / f"v{version_num}.json"
                    candidate_metadata = folder / f"v{version_num}_metadata.json"
                    if candidate_path.exists():
                        load_path = candidate_path
                        metadata_path = candidate_metadata
                        break

                if load_path is None:
                    raise FileNotFoundError(f"No versions found in {folder}")

            # Load from the determined path
            super().load(load_path)

            # Load metadata score
            if metadata_path and metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    program_score = metadata.get("score")
                except (json.JSONDecodeError, KeyError):
                    program_score = None
            else:
                program_score = None
        else:
            raise ValueError("Either 'path' or 'optimizer' must be provided")

        # set score
        self.score = program_score
        # Reset other state
        self.optimized_program = None
        self.optimizer = None

    def load_best(self, optimizer: Teleprompter | None = None):
        """
        Load the best performing model based on score from metadata files.

        Args:
            optimizer: Optional optimizer to filter by. If provided, only models
                      from that optimizer will be considered.

        Returns:
            TimeStructureBlueprintGenerator: Self with the best model loaded
        """
        model_name = str(self.lm.model).split("/")[-1]
        script_dir = Path(__file__).parent
        base_path = script_dir / "optim" / model_name

        if not base_path.exists():
            raise FileNotFoundError(f"Model folder not found: {base_path}")

        best_score = -1
        best_path = None
        best_metadata_path = None
        best_optimizer_name = None

        if optimizer is not None:
            # Filter to specific optimizer
            optimizer_name = optimizer.__class__.__name__
            optimizer_folders = [base_path / optimizer_name]
            if not optimizer_folders[0].exists():
                raise FileNotFoundError(
                    f"Optimizer folder not found: {optimizer_folders[0]}"
                )
        else:
            # Check all optimizer folders
            optimizer_folders = [
                folder for folder in base_path.iterdir() if folder.is_dir()
            ]

        for optimizer_folder in optimizer_folders:
            if not optimizer_folder.exists():
                continue

            # Find all metadata files in this optimizer folder
            for metadata_file in optimizer_folder.glob("v*_metadata.json"):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    score = metadata.get("score")
                    if score is not None and score > best_score:
                        best_score = score
                        best_metadata_path = metadata_file
                        # Get corresponding model file
                        version = metadata_file.stem.replace("_metadata", "")
                        best_path = optimizer_folder / f"{version}.json"
                        best_optimizer_name = optimizer_folder.name
                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    # Skip invalid metadata files
                    continue

        if best_path is None or not best_path.exists():
            optimizer_info = (
                f" for optimizer {optimizer.__class__.__name__}" if optimizer else ""
            )
            raise FileNotFoundError(f"No valid optimized models found{optimizer_info}")

        print(
            f"🏆 Loading best model: {best_optimizer_name} with score {best_score:.4f}"
        )
        print(f"📁 Path: {best_path}")

        # Delegate to the main load method
        self.load(path=best_path)



    ##### Optimizers #####

    def BootstrapFewShot(
        self,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 2,
        max_rounds: int = 1,
        metric_threshold: Optional[float] = None,
        teacher: Optional[dspy.LM] = None,
    ):
        """
        Optimize the program using Bootstrap Few-Shot and return the optimized program with metadata.

        Returns:
            Optimized program with injected optimizer metadata
        """
        self.ensure_metric()
        self.ensure_trainset()
        self.ensure_valset()
        optimizer = dspy.BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            metric_threshold=metric_threshold,
            # teacher_settings={"lm": teacher},
        )

        if teacher:
            teacher_program = self.deepcopy()
            teacher_program.lm = teacher
        else:
            teacher_program = None

        optimized_program: Program = optimizer.compile(
            student=self, trainset=self.trainset, teacher=teacher_program
        )

        optimized_program.score = None
        optimized_program.optimizer = optimizer
        self.optimized_program = optimized_program

        return optimized_program

    def MIPROv2(
        self,
        auto: Literal["light", "medium", "heavy"] = "medium",
        num_threads: int = 10,
        teacher: Optional[dspy.LM] = None,
    ):
        """
        Optimize the program using MIPROv2 and return the optimized program with metadata.
        """
        self.ensure_metric()
        self.ensure_trainset()
        self.ensure_valset()
        # Initialize optimizer
        teleprompter = dspy.MIPROv2(
            metric=self.metric,
            auto=auto,  # Can choose between light, medium, and heavy optimization runs
            seed=self.seed,
            num_threads=num_threads,
            teacher_settings={"lm": teacher},
        )

        optimized_program: Program = teleprompter.compile(
            student=self,
            trainset=self.trainset,
            valset=self.valset,
        )

        optimized_program.score = None
        optimized_program.optimizer = teleprompter
        self.optimized_program = optimized_program
        return optimized_program

    def SIMBA(
        self,
        bsize: int = 32,
        num_candidates: int = 6,
        max_steps: int = 8,
        num_threads: Optional[int] = None,
    ):
        self.ensure_metric()
        self.ensure_trainset()
        self.ensure_valset()
        optimizer = dspy.SIMBA(
            metric=self.metric,
            bsize=bsize,
            num_candidates=num_candidates,
            max_steps=max_steps,
            num_threads=num_threads,
        )
        optimized_program: Program = optimizer.compile(
            student=self,
            trainset=self.trainset,
            seed=self.seed,
        )
        optimized_program.optimizer = optimizer
        self.optimized_program = optimized_program
        return optimized_program


    def GEPA(
        self,
        metric,  # Required: GEPA needs a 5-argument metric (gold, pred, trace, pred_name, pred_trace)
        auto: Optional[Literal["light", "medium", "heavy"]] = None,
        reflection_lm: Optional[dspy.LM] = None,
        num_threads: Optional[int] = None,
        teacher: Optional[dspy.LM] = None,
    ):
        # GEPA uses its own metric, so we skip the standard metric check
        self.ensure_trainset()
        self.ensure_valset()
        optimizer = dspy.GEPA(
            metric=metric,
            auto=auto,
            reflection_lm=reflection_lm,
            num_threads=num_threads,
        )

        if teacher:
            teacher_program = self.deepcopy()
            teacher_program.lm = teacher
        else:
            teacher_program = None
        
        optimized_program: Program = optimizer.compile(
            student=self,
            trainset=self.trainset,
            valset=self.valset,
            teacher=teacher_program,
        )
        optimized_program.optimizer = optimizer
        self.optimized_program = optimized_program
        return optimized_program
