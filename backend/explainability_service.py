import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from backend.model_service import model_service


class ExplainabilityService:
    def __init__(self):
        self.model = model_service.model
        self.processor = model_service.audio_processor
        self.class_names = model_service.class_names

    def _segment_probabilities(self, segments):
        self.model.eval()
        tensors = []
        for segment in segments:
            mel = self.processor.compute_mel_spectrogram(segment)
            tensors.append(mel)
        batch = torch.tensor(
            np.stack(tensors)[:, np.newaxis, :, :],
            dtype=torch.float32,
            device=model_service.device,
        )
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy(), tensors

    def generate_gradcam(self, file_path: str, target_genre: str | None = None):
        # Use the exact same preprocessing and overall prediction used by /predict.
        overall = model_service.predict_file(file_path)
        predicted_genre = str(target_genre or overall["genre"])

        # Resolve the requested genre against the production class names
        # without changing the canonical class-name casing.
        class_lookup = {
            name.lower(): name
            for name in self.class_names
        }

        canonical_genre = class_lookup.get(predicted_genre.lower())

        if canonical_genre is None:
            canonical_genre = overall["genre"]

        predicted_genre = canonical_genre
        target_index = self.class_names.index(predicted_genre)

        y = self.processor.load_and_preprocess_raw_audio(file_path)
        segments = self.processor.extract_segments(y)
        if not segments:
            raise ValueError("Could not extract any audio segments.")

        probabilities, mels = self._segment_probabilities(segments)
        # Explain a meaningful segment that contributes strongly to the SAME final genre.
        rms = np.array([float(np.sqrt(np.mean(np.square(seg))) + 1e-12) for seg in segments])
        max_rms = float(rms.max()) if len(rms) else 1.0
        meaningful = rms >= max_rms * 0.10
        candidates = np.where(meaningful)[0] if np.any(meaningful) else np.arange(len(segments))
        segment_index = int(candidates[np.argmax(probabilities[candidates, target_index])])
        mel = mels[segment_index]

        input_tensor = torch.tensor(
            mel[np.newaxis, np.newaxis, :, :],
            dtype=torch.float32,
            device=model_service.device,
            requires_grad=True,
        )

        activations = None
        gradients = None

        def forward_hook(module, module_input, output):
            nonlocal activations
            activations = output

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        target_layer = self.model.conv4[0]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = self.model(input_tensor)
                target_score = logits[0, target_index]
                target_score.backward()

            if activations is None or gradients is None:
                raise RuntimeError("Could not capture Grad-CAM activations.")

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=mel.shape, mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max - cam_min > 1e-8 else np.zeros_like(cam)

        overall_confidence = float(overall["probabilities"][predicted_genre])

        fig, ax = plt.subplots(figsize=(10, 3.4), dpi=140)
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", interpolation="nearest", extent=[0, self.processor.segment_duration, 0, self.processor.n_mels])
        ax.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=np.clip(cam * 0.55, 0, 0.55), interpolation="bilinear", extent=[0, self.processor.segment_duration, 0, self.processor.n_mels])
        ax.set_xlabel("Time within selected segment (s)", fontsize=9, labelpad=8)
        ax.set_ylabel("Mel frequency", fontsize=9, labelpad=8)
        ax.tick_params(axis="both", labelsize=8, length=0, colors="#AAA4A0")
        ax.set_title(f"Grad-CAM | {predicted_genre.title()} | {overall_confidence * 100:.2f}%", fontsize=11, fontweight="medium", pad=12, color="#FFFFFF")
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.patch.set_facecolor("#171717")
        ax.set_facecolor("#171717")
        ax.xaxis.label.set_color("#DDD7D3")
        ax.yaxis.label.set_color("#DDD7D3")
        fig.tight_layout(pad=1.4)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        return {
            "image": buffer.getvalue(),
            "genre": predicted_genre,
            "confidence": overall_confidence,
            "segment_index": segment_index,
            "segment_duration": self.processor.segment_duration,
        }


explainability_service = ExplainabilityService()
