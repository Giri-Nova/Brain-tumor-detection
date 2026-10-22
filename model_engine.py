import os
import io
import time
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import matplotlib
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_cnn_final.keras")
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

COLORMAPS = {
    "jet": matplotlib.colormaps["jet"],
    "turbo": matplotlib.colormaps["turbo"],
    "inferno": matplotlib.colormaps["inferno"],
    "plasma": matplotlib.colormaps["plasma"],
    "viridis": matplotlib.colormaps["viridis"]
}

class TumorModelEngine:
    def __init__(self):
        self.model = None
        self.target_layer_name = "conv2d_3"
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading neural network model from: {MODEL_PATH}")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            # Warm up model with dummy input
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.model(dummy)
            self.is_loaded = True
            print("Model successfully loaded and initialized!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False

    def preprocess_image(self, pil_image: Image.Image, enhance_contrast: bool = False):
        img_rgb = pil_image.convert("RGB")
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img_rgb)
            img_rgb = enhancer.enhance(1.25)
        
        resized = img_rgb.resize((224, 224))
        arr = np.array(resized, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0), img_rgb

    def compute_gradcam(self, input_tensor, target_class_idx=None):
        if not self.is_loaded or self.model is None:
            return np.zeros((224, 224), dtype=np.float32)
        
        try:
            with tf.GradientTape() as tape:
                x = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
                conv_out = None
                remaining_layers = []
                found = False

                for layer in self.model.layers:
                    if not found:
                        x = layer(x)
                        if layer.name == self.target_layer_name:
                            conv_out = x
                            found = True
                    else:
                        remaining_layers.append(layer)

                tape.watch(conv_out)
                y = conv_out
                for layer in remaining_layers:
                    y = layer(y)

                if target_class_idx is None:
                    target_class_idx = tf.argmax(y[0])

                loss = y[:, target_class_idx]

            grads = tape.gradient(loss, conv_out)
            if grads is None:
                return np.zeros((224, 224), dtype=np.float32)

            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_out[0]), axis=-1)
            cam = np.maximum(cam.numpy(), 0)
            max_val = np.max(cam)
            if max_val > 1e-7:
                cam = cam / max_val
            else:
                cam = np.zeros_like(cam)

            # Resize 24x24 CAM back to 224x224
            cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC)
            cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0
            return cam_arr
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return np.zeros((224, 224), dtype=np.float32)

    def generate_heatmap_overlay(self, original_rgb: Image.Image, cam_array: np.ndarray, colormap_name="jet", alpha=0.5):
        orig_resized = original_rgb.resize((224, 224))
        orig_np = np.array(orig_resized, dtype=np.float32) / 255.0

        cmap = COLORMAPS.get(colormap_name, COLORMAPS["jet"])
        colored_cam = cmap(cam_array)[:, :, :3]

        # Blend
        overlay_np = (1.0 - alpha) * orig_np + alpha * colored_cam
        overlay_np = np.clip(overlay_np, 0.0, 1.0)
        overlay_pil = Image.fromarray((overlay_np * 255).astype(np.uint8))

        # Colored CAM alone with transparent or black background
        heatmap_pil = Image.fromarray((colored_cam * 255).astype(np.uint8))

        return overlay_pil, heatmap_pil

    def find_roi_bounding_box(self, cam_array: np.ndarray, threshold_ratio=0.55):
        if np.max(cam_array) < 0.15:
            return None
        
        threshold = np.max(cam_array) * threshold_ratio
        y_indices, x_indices = np.where(cam_array >= threshold)

        if len(y_indices) < 20: # too small/noise
            return None

        h, w = cam_array.shape
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))

        # Add a subtle margin
        margin_y = max(4, int((ymax - ymin) * 0.1))
        margin_x = max(4, int((xmax - xmin) * 0.1))

        ymin = max(0, ymin - margin_y)
        ymax = min(h, ymax + margin_y)
        xmin = max(0, xmin - margin_x)
        xmax = min(w, xmax + margin_x)

        cx = int(np.mean(x_indices))
        cy = int(np.mean(y_indices))

        # Convert to percentages for responsive frontend CSS placement
        return {
            "ymin_pct": round((ymin / h) * 100, 2),
            "ymax_pct": round((ymax / h) * 100, 2),
            "xmin_pct": round((xmin / w) * 100, 2),
            "xmax_pct": round((xmax / w) * 100, 2),
            "width_pct": round(((xmax - xmin) / w) * 100, 2),
            "height_pct": round(((ymax - ymin) / h) * 100, 2),
            "centroid_x_pct": round((cx / w) * 100, 2),
            "centroid_y_pct": round((cy / h) * 100, 2),
            "lesion_area_pct": round((len(y_indices) / (h * w)) * 100, 2),
            "pixel_box": [xmin, ymin, xmax, ymax]
        }

    def predict(self, pil_image: Image.Image, colormap_name="jet", enhance_contrast=False):
        start_time = time.time()
        input_tensor, orig_rgb = self.preprocess_image(pil_image, enhance_contrast)

        preds = self.model.predict(input_tensor, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx])

        # Compute Grad-CAM
        cam_arr = self.compute_gradcam(input_tensor, target_class_idx=top_idx)
        overlay_pil, heatmap_pil = self.generate_heatmap_overlay(orig_rgb, cam_arr, colormap_name=colormap_name)

        # Lesion region
        has_tumor = (predicted_class != "No Tumor")
        roi_box = self.find_roi_bounding_box(cam_arr) if has_tumor else None

        # Base64 encodings
        def to_b64(img: Image.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        # Class probabilities dictionary
        probabilities = [
            {
                "class_name": CLASS_NAMES[i],
                "probability": float(preds[i]),
                "percentage": round(float(preds[i]) * 100, 2)
            }
            for i in range(len(CLASS_NAMES))
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)

        # Clinical recommendation details
        clinical_details = self._get_clinical_details(predicted_class, confidence, roi_box)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "is_tumor_detected": has_tumor,
            "probabilities": probabilities,
            "inference_time_ms": elapsed_ms,
            "heatmap_data_uri": to_b64(heatmap_pil),
            "overlay_data_uri": to_b64(overlay_pil),
            "roi_bounding_box": roi_box,
            "clinical_details": clinical_details
        }

    def _get_clinical_details(self, predicted_class: str, confidence: float, roi_box):
        if predicted_class == "Glioma":
            severity = "High Alert" if confidence > 0.8 else "Moderate Suspicion"
            return {
                "category": "Intra-axial Glial Neoplasm",
                "severity": severity,
                "urgency": "Immediate Neurosurgical Oncology Consultation",
                "pathology_notes": "Lesion characteristics correspond to parenchymal glial neoplasm. Typical findings include hyperintense signals on T2/FLAIR with irregular peripheral contrast enhancement and surrounding vasogenic edema.",
                "recommended_followup": "High-resolution 3T MRI with perfusion, spectroscopy (MRS), and stereotactic biopsy evaluation.",
                "anatomical_location": "Cerebral white matter / Hemispheric parenchyma"
            }
        elif predicted_class == "Meningioma":
            severity = "Elevated Alert" if confidence > 0.8 else "Moderate Suspicion"
            return {
                "category": "Extra-axial Dural-based Tumor",
                "severity": severity,
                "urgency": "Neurosurgical Clinic Referral",
                "pathology_notes": "Well-circumscribed extra-axial lesion arising from arachnoid cap cells, demonstrating homogeneous avid enhancement and possible dural tail sign.",
                "recommended_followup": "Contrast-enhanced volumetric 3D MRI, assessment of adjacent venous sinus patency, and neurosurgical resection planning.",
                "anatomical_location": "Convexity / Falx / Parasagittal dura"
            }
        elif predicted_class == "Pituitary":
            severity = "Specialized Alert"
            return {
                "category": "Sellar / Suprasellar Macroadenoma",
                "severity": severity,
                "urgency": "Endocrinology & Neuro-ophthalmology Workup",
                "pathology_notes": "Expansion of the sella turcica with soft tissue mass potentially compressing the optic chiasm. Evaluation of anterior pituitary hormones advised.",
                "recommended_followup": "Dedicated thin-slice pituitary protocol MRI, visual field perimetry test, serum prolactin, IGF-1, cortisol panel.",
                "anatomical_location": "Sella turcica / Suprasellar cistern"
            }
        else: # No Tumor
            return {
                "category": "Unremarkable Neuroimaging",
                "severity": "Normal",
                "urgency": "Routine Clinical Follow-up",
                "pathology_notes": "No hyperintense mass lesions, pathological enhancement, abnormal midline shift, or mass effect identified on the presented slices.",
                "recommended_followup": "Correlate with clinical symptomatology; repeat imaging only if new focal neurological deficits emerge.",
                "anatomical_location": "Symmetrical cerebral architecture"
            }

# Global singleton
engine = TumorModelEngine()
