import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_dotenv_if_exists(env_path: Path) -> None:
	if not env_path.exists():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key:
			os.environ.setdefault(key, value)


def configure_kaggle_environment() -> None:
	# As credenciais do Kaggle estão no .env.
	load_dotenv_if_exists(Path(".env"))

	if os.getenv("KAGGLE_API_KEY") and not os.getenv("KAGGLE_KEY"):
		os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_KEY"]

	if os.getenv("KAGGLE_API_USERNAME") and not os.getenv("KAGGLE_USERNAME"):
		os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_API_USERNAME"]

	has_env_auth = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
	has_file_auth = Path.home().joinpath(".kaggle", "kaggle.json").exists()

	if not has_env_auth and not has_file_auth:
		raise RuntimeError(
			"Autenticacao Kaggle nao configurada. Defina no .env: "
			"KAGGLE_USERNAME e KAGGLE_API_KEY (ou KAGGLE_KEY), "
			"ou crie ~/.kaggle/kaggle.json."
		)

# Encontra o melhor dispositivo disponível, priorizando CUDA > MPS > CPU.
def resolve_device(device_arg: str) -> torch.device:
	device_arg = device_arg.lower()
	if device_arg not in {"auto", "cpu", "cuda", "mps"}:
		raise ValueError("--device deve ser um de: auto, cpu, cuda, mps")

	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA nao esta disponivel nesta maquina.")
		return torch.device("cuda")
	if device_arg == "mps":
		if not torch.backends.mps.is_available():
			raise RuntimeError("MPS nao esta disponivel nesta maquina.")
		return torch.device("mps")

	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def build_dataloader_kwargs(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
	use_pin_memory = device.type == "cuda"
	num_workers = max(0, args.num_workers)
	kwargs: Dict[str, object] = {
		"num_workers": num_workers,
		"pin_memory": use_pin_memory,
	}
	if num_workers > 0:
		kwargs["persistent_workers"] = True
	return kwargs


@dataclass
class SplitFrames:
	train: pd.DataFrame
	val: pd.DataFrame
	test: pd.DataFrame


class HAM10000Dataset(Dataset):
	def __init__(
		self,
		frame: pd.DataFrame,
		image_map: Dict[str, Path],
		class_to_idx: Dict[str, int],
		transform: transforms.Compose,
	) -> None:
		self.frame = frame.reset_index(drop=True)
		self.image_map = image_map
		self.class_to_idx = class_to_idx
		self.transform = transform

	def __len__(self) -> int:
		return len(self.frame)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		row = self.frame.iloc[idx]
		image_id = row["image_id"]
		image_path = self.image_map[image_id]
		image = Image.open(image_path).convert("RGB")
		label = self.class_to_idx[row["dx"]]
		return self.transform(image), label


def ensure_dataset(root_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Path]]:
	root_dir.mkdir(parents=True, exist_ok=True)
	metadata_file = find_metadata_file(root_dir)

	if metadata_file is None:
		configure_kaggle_environment()

		try:
			from kaggle.api.kaggle_api_extended import KaggleApi
		except ImportError as exc:
			raise RuntimeError(
				"Pacote kaggle nao encontrado. Instale com: pip install kaggle"
			) from exc

		print("Dataset nao encontrado localmente. Baixando via Kaggle API...")
		api = KaggleApi()
		api.authenticate()
		api.dataset_download_files(
			"kmader/skin-cancer-mnist-ham10000", path=str(root_dir), unzip=True
		)
		metadata_file = find_metadata_file(root_dir)

	if metadata_file is None:
		raise FileNotFoundError(
			"Nao foi possivel localizar HAM10000_metadata.csv apos download."
		)

	dataset_root = metadata_file.parent
	frame = pd.read_csv(metadata_file)
	image_paths = sorted(dataset_root.glob("HAM10000_images_part_*/*.jpg"))
	if not image_paths:
		image_paths = sorted(dataset_root.rglob("*.jpg"))

	image_map = {img_path.stem: img_path for img_path in image_paths}
	missing = frame.loc[~frame["image_id"].isin(image_map.keys()), "image_id"]
	if not missing.empty:
		raise RuntimeError(
			f"Foram encontradas imagens faltantes no dataset: {missing.iloc[:5].tolist()}"
		)

	return frame[["image_id", "dx"]].copy(), image_map


def find_metadata_file(root_dir: Path) -> Path | None:
	candidates = list(root_dir.rglob("HAM10000_metadata.csv"))
	return candidates[0] if candidates else None


def build_splits(
	frame: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> SplitFrames:
	train_df, test_df = train_test_split(
		frame,
		test_size=test_size,
		random_state=seed,
		stratify=frame["dx"],
	)
	adjusted_val = val_size / (1.0 - test_size)
	train_df, val_df = train_test_split(
		train_df,
		test_size=adjusted_val,
		random_state=seed,
		stratify=train_df["dx"],
	)
	return SplitFrames(train=train_df, val=val_df, test=test_df)


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
	train_tf = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	eval_tf = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return train_tf, eval_tf


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
	weights = models.EfficientNet_B4_Weights.DEFAULT
	model = models.efficientnet_b4(weights=weights)

	if freeze_backbone:
		for param in model.features.parameters():
			param.requires_grad = False

	in_features = model.classifier[1].in_features
	model.classifier[1] = nn.Linear(in_features, num_classes)
	return model


def run_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer | None,
	device: torch.device,
	phase_name: str,
	log_interval: int,
) -> Tuple[float, List[int], List[int], np.ndarray]:
	is_train = optimizer is not None
	model.train() if is_train else model.eval()

	total_loss = 0.0
	all_true: List[int] = []
	all_pred: List[int] = []
	all_probs: List[np.ndarray] = []
	total_batches = len(loader)
	start_time = time.time()

	with torch.set_grad_enabled(is_train):
		for batch_idx, (inputs, targets) in enumerate(loader, start=1):
			inputs = inputs.to(device)
			targets = targets.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, targets)

			if is_train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			probs = torch.softmax(outputs, dim=1)
			preds = torch.argmax(probs, dim=1)

			total_loss += loss.item() * inputs.size(0)
			all_true.extend(targets.detach().cpu().numpy().tolist())
			all_pred.extend(preds.detach().cpu().numpy().tolist())
			all_probs.append(probs.detach().cpu().numpy())

			if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == total_batches):
				elapsed = time.time() - start_time
				avg_time = elapsed / batch_idx
				remaining_batches = total_batches - batch_idx
				eta = remaining_batches * avg_time
				print(
					f"[{phase_name}] batch {batch_idx}/{total_batches} | "
					f"loss={loss.item():.4f} | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
					flush=True,
				)

	avg_loss = total_loss / len(loader.dataset)
	probs_array = np.concatenate(all_probs, axis=0)
	return avg_loss, all_true, all_pred, probs_array


def compute_metrics(y_true: List[int], y_pred: List[int], probs: np.ndarray) -> Dict[str, float]:
	labels = sorted(set(y_true))
	y_true_bin = label_binarize(y_true, classes=labels)
	metrics = {
		"accuracy": accuracy_score(y_true, y_pred),
		"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
		"precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
		"precision_weighted": precision_score(
			y_true, y_pred, average="weighted", zero_division=0
		),
		"recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
		"recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
		"f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
		"log_loss": log_loss(y_true, np.clip(probs, 1e-8, 1.0 - 1e-8)),
	}

	try:
		metrics["roc_auc_ovr_macro"] = roc_auc_score(
			y_true_bin,
			probs,
			average="macro",
			multi_class="ovr",
		)
	except ValueError:
		metrics["roc_auc_ovr_macro"] = float("nan")

	return metrics


def evaluate_split(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	idx_to_class: Dict[int, str],
	log_interval: int,
	phase_name: str,
) -> Dict[str, object]:
	loss, y_true, y_pred, probs = run_epoch(
		model,
		loader,
		criterion,
		None,
		device,
		phase_name=phase_name,
		log_interval=log_interval,
	)
	metrics = compute_metrics(y_true, y_pred, probs)
	metrics["loss"] = loss

	target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
	report = classification_report(
		y_true,
		y_pred,
		target_names=target_names,
		zero_division=0,
		output_dict=True,
	)
	cm = confusion_matrix(y_true, y_pred).tolist()

	return {
		"metrics": metrics,
		"classification_report": report,
		"confusion_matrix": cm,
	}


def unfreeze_backbone(model: nn.Module) -> None:
	for param in model.features.parameters():
		param.requires_grad = True


def train(args: argparse.Namespace) -> None:
	set_seed(args.seed)
	device = resolve_device(args.device)
	print(f"Dispositivo em uso: {device}")
	print("Iniciando preparacao de dados...", flush=True)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	frame, image_map = ensure_dataset(Path(args.dataset_dir))
	splits = build_splits(frame, args.val_size, args.test_size, args.seed)
	print(
        
		f"Dataset pronto. Total={len(frame)} | Train={len(splits.train)} | "
		f"Val={len(splits.val)} | Test={len(splits.test)}",
		flush=True,
	)

	classes = sorted(frame["dx"].unique().tolist())
	class_to_idx = {name: idx for idx, name in enumerate(classes)}
	idx_to_class = {idx: name for name, idx in class_to_idx.items()}

	with open(output_dir / "class_to_idx.json", "w", encoding="utf-8") as fp:
		json.dump(class_to_idx, fp, indent=2, ensure_ascii=False)

	train_tf, eval_tf = make_transforms(args.image_size)

	datasets = {
		"train": HAM10000Dataset(splits.train, image_map, class_to_idx, train_tf),
		"val": HAM10000Dataset(splits.val, image_map, class_to_idx, eval_tf),
		"test": HAM10000Dataset(splits.test, image_map, class_to_idx, eval_tf),
	}
	dl_kwargs = build_dataloader_kwargs(args, device)
	print(
		f"DataLoader configurado com num_workers={dl_kwargs['num_workers']} "
		f"e pin_memory={dl_kwargs['pin_memory']}",
		flush=True,
	)
	loaders = {
		"train": DataLoader(
			datasets["train"],
			batch_size=args.batch_size,
			shuffle=True,
			**dl_kwargs,
		),
		"val": DataLoader(
			datasets["val"],
			batch_size=args.batch_size,
			shuffle=False,
			**dl_kwargs,
		),
		"test": DataLoader(
			datasets["test"],
			batch_size=args.batch_size,
			shuffle=False,
			**dl_kwargs,
		),
	}

	model = build_model(num_classes=len(classes), freeze_backbone=True).to(device)
	criterion = nn.CrossEntropyLoss()
	print("Modelo EfficientNet-B4 carregado. Iniciando treinamento...", flush=True)

	best_val_f1 = -1.0
	best_model_path = output_dir / "best_efficientnet_b4_ham10000.pth"
	history: List[Dict[str, float]] = []

	def train_stage(epochs: int, lr: float, stage_name: str) -> None:
		nonlocal best_val_f1
		optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
		scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
		print(f"\n== Etapa {stage_name}: {epochs} epocas, lr={lr} ==", flush=True)

		for epoch in range(1, epochs + 1):
			print(f"\n[{stage_name}] Epoca {epoch}/{epochs} iniciada", flush=True)
			train_loss, y_t, y_p, probs = run_epoch(
				model,
				loaders["train"],
				criterion,
				optimizer,
				device,
				phase_name=f"{stage_name}-train-ep{epoch}",
				log_interval=args.log_interval,
			)
			train_metrics = compute_metrics(y_t, y_p, probs)

			val_result = evaluate_split(
				model,
				loaders["val"],
				criterion,
				device,
				idx_to_class,
				log_interval=args.log_interval,
				phase_name=f"{stage_name}-val-ep{epoch}",
			)
			val_metrics = val_result["metrics"]

			scheduler.step()

			row = {
				"stage": stage_name,
				"epoch": epoch,
				"train_loss": train_loss,
				"train_f1_macro": train_metrics["f1_macro"],
				"train_accuracy": train_metrics["accuracy"],
				"val_loss": val_metrics["loss"],
				"val_f1_macro": val_metrics["f1_macro"],
				"val_accuracy": val_metrics["accuracy"],
			}
			history.append(row)

			print(
				f"[{stage_name}] Epoca {epoch}/{epochs} | "
				f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
				f"train_f1={train_metrics['f1_macro']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
			)

			if val_metrics["f1_macro"] > best_val_f1:
				best_val_f1 = val_metrics["f1_macro"]
				torch.save(
					{
						"model_state_dict": model.state_dict(),
						"class_to_idx": class_to_idx,
						"image_size": args.image_size,
						"architecture": "efficientnet_b4",
					},
					best_model_path,
				)

	if args.epochs_head > 0:
		train_stage(args.epochs_head, args.lr_head, "head")

	if args.epochs_finetune > 0:
		unfreeze_backbone(model)
		train_stage(args.epochs_finetune, args.lr_finetune, "finetune")

	checkpoint = torch.load(best_model_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])

	full_results = {
		"splits": {
			"train_size": len(datasets["train"]),
			"val_size": len(datasets["val"]),
			"test_size": len(datasets["test"]),
			"classes": classes,
		},
		"history": history,
		"evaluation": {},
	}

	for split_name in ["train", "val", "test"]:
		print(f"\nAvaliando split: {split_name}", flush=True)
		result = evaluate_split(
			model,
			loaders[split_name],
			criterion,
			device,
			idx_to_class,
			log_interval=args.log_interval,
			phase_name=f"eval-{split_name}",
		)
		full_results["evaluation"][split_name] = result
		print(f"\nMetrica principal ({split_name}):")
		print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))

	history_df = pd.DataFrame(history)
	history_df.to_csv(output_dir / "training_history.csv", index=False)
	with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as fp:
		json.dump(full_results, fp, indent=2, ensure_ascii=False)

	print("\nTreinamento finalizado com sucesso.")
	print(f"Modelo salvo em: {best_model_path}")
	print(f"Metricas em: {output_dir / 'metrics_summary.json'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Treino de classificador de cancer de pele com EfficientNet-B4"
	)
	parser.add_argument("--dataset-dir", type=str, default="./data")
	parser.add_argument("--output-dir", type=str, default="./artifacts")
	parser.add_argument("--image-size", type=int, default=380)
	parser.add_argument("--batch-size", type=int, default=16)
	default_workers = 2 if sys.platform == "darwin" else max(1, os.cpu_count() // 2)
	parser.add_argument("--num-workers", type=int, default=default_workers)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		help="Dispositivo: auto, cpu, mps ou cuda",
	)
	parser.add_argument("--epochs-head", type=int, default=5)
	parser.add_argument("--epochs-finetune", type=int, default=10)
	parser.add_argument("--lr-head", type=float, default=1e-3)
	parser.add_argument("--lr-finetune", type=float, default=3e-5)
	parser.add_argument("--val-size", type=float, default=0.15)
	parser.add_argument("--test-size", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--log-interval",
		type=int,
		default=20,
		help="Quantidade de batches entre logs de progresso (0 desativa)",
	)
	return parser.parse_args()


if __name__ == "__main__":
	arguments = parse_args()
	train(arguments)
