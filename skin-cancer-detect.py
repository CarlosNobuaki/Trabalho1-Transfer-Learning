import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def resolve_device(device_arg: str) -> torch.device:
	device_arg = device_arg.lower()
	if device_arg not in {"auto", "cpu", "cuda", "mps"}:
		raise ValueError("--device deve ser um de: auto, cpu, cuda ou mps")

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

	# Prioriza MPS em Apple Silicon e CUDA em outros cenarios.
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
	model = models.efficientnet_b4(weights=None)
	in_features = model.classifier[1].in_features
	model.classifier[1] = nn.Linear(in_features, num_classes)
	return model


def get_transform(image_size: int) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def predict_image(
	image_path: Path,
	model_path: Path,
	class_map_path: Path | None,
	top_k: int,
	device_arg: str,
) -> None:
	device = resolve_device(device_arg)
	print(f"Dispositivo em uso: {device}")
	checkpoint = torch.load(model_path, map_location=device)

	class_to_idx = checkpoint.get("class_to_idx")
	if class_to_idx is None:
		if class_map_path is None:
			raise ValueError(
				"O checkpoint nao possui class_to_idx e nenhum --class-map foi informado."
			)
		with open(class_map_path, "r", encoding="utf-8") as fp:
			class_to_idx = json.load(fp)

	image_size = checkpoint.get("image_size", 380)
	idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

	model = build_model(num_classes=len(class_to_idx)).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	transform = get_transform(image_size)
	image = Image.open(image_path).convert("RGB")
	tensor = transform(image).unsqueeze(0).to(device)

	with torch.no_grad():
		logits = model(tensor)
		probs = torch.softmax(logits, dim=1).squeeze(0)

	top_k = min(top_k, len(class_to_idx))
	values, indices = torch.topk(probs, k=top_k)

	print(f"Imagem: {image_path}")
	print("Top predicoes:")
	for rank, (prob, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
		class_name = idx_to_class[idx]
		print(f"{rank}. {class_name} -> {prob * 100:.2f}%")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Classificar imagem de pele usando modelo EfficientNet-B4 treinado"
	)
	parser.add_argument("--image", type=str, required=True, help="Caminho da imagem")
	parser.add_argument(
		"--model",
		type=str,
		default="./artifacts/best_efficientnet_b4_ham10000.pth",
		help="Checkpoint gerado no treinamento",
	)
	parser.add_argument(
		"--class-map",
		type=str,
		default=None,
		help="Arquivo class_to_idx.json (opcional se o checkpoint ja tiver este mapa)",
	)
	parser.add_argument("--top-k", type=int, default=3)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		help="Dispositivo: auto, mps, cpu ou cuda",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	predict_image(
		image_path=Path(args.image),
		model_path=Path(args.model),
		class_map_path=Path(args.class_map) if args.class_map else None,
		top_k=args.top_k,
		device_arg=args.device,
	)
