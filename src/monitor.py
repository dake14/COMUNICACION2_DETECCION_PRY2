"""Monitor integrado de cinco salidas y procesador CLI de señales desconocidas (Nivel 1).

Proporciona las cinco salidas obligatorias del proyecto bajo contrato explícito:
1. Detección de presencia (calibrada por referencia y controlada por archivo).
2. Densidad espectral de potencia (PSD) en Hz o ciclos/muestra y ancho de banda.
3. Estimación de SNR condicionada a detección y referencia compatible.
4. Tipo de modulación reconocida en alcance cerrado de seis clases.
5. Confianza de la decisión final, calibrada solo cuando modelo/agregación coinciden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.conventional import CLASSES, _validate_artifact
from src.detection import detect_energy, detect_energy_file, energy_statistic
from src.features import FEATURE_VERSION
from src.snr_estimation import calibrate_noise_reference, estimate_snr
from src.spectrum import estimate_spectrum


DEFAULT_SVM_MODEL = Path("results/fase5/conventional_256.joblib")
DEFAULT_CALIBRATOR = Path("results/nivel1_monitor/monitor_calibrators.joblib")
DEFAULT_POLICY = Path("results/adaptive/policy.joblib")

SUPPORTED_EXTENSIONS = (".c64", ".bin", ".npy", ".npz", ".csv")
EXPECTED_MODEL_FS_HZ = 8000.0
EXPECTED_MODEL_SPS = 8
EXPECTED_MODEL_SYMBOL_RATE_HZ = 1000.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} debe ser un número real finito y positivo.")
    return float(value)


def _validate_nonnegative_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} debe ser un número real finito y no negativo.")
    return float(value)


def _extract_iq_vector(arr: np.ndarray, source_desc: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        raise ValueError(f"El arreglo en {source_desc} está vacío.")
    if arr.dtype == object or not (np.issubdtype(arr.dtype, np.number)):
        raise ValueError(f"Tipo de dato no numérico en {source_desc}: {arr.dtype}.")
    if arr.ndim == 1:
        if not np.iscomplexobj(arr):
            raise ValueError(
                f"Arreglo real 1D ambiguo sin componente Q en {source_desc}; use complejo o matriz Nx2 (I, Q)."
            )
        samples = arr
    elif arr.ndim == 2:
        if np.iscomplexobj(arr):
            if arr.shape[0] == 1 or arr.shape[1] == 1:
                samples = arr.reshape(-1)
            else:
                raise ValueError(
                    f"Lote 2D complejo de forma {arr.shape} en {source_desc}; proporcione una sola observación."
                )
        else:
            if arr.shape == (2, 2):
                raise ValueError(
                    f"Matriz real 2x2 ambigua en {source_desc} (ejes I/Q indistinguibles de 2 muestras)."
                )
            if arr.shape[1] == 2:
                samples = arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
            elif arr.shape[0] == 2:
                samples = arr[0, :].astype(np.float64) + 1j * arr[1, :].astype(np.float64)
            else:
                raise ValueError(
                    f"Matriz real de forma {arr.shape} inválida en {source_desc}; se esperaba (N, 2) o (2, N)."
                )
    else:
        raise ValueError(f"Dimensionalidad {arr.ndim}D no soportada en {source_desc}; se requiere 1D o Nx2.")

    if samples.size < 2:
        raise ValueError(f"La señal en {source_desc} tiene menos de dos muestras ({samples.size}).")
    if not np.isfinite(samples).all():
        raise ValueError(f"La señal en {source_desc} contiene valores infinitos o NaN.")
    return np.asarray(samples, dtype=np.complex128)


def _load_sidecar_metadata(path: Path) -> dict:
    sidecar = path.with_name(path.name + ".meta.json")
    if not sidecar.is_file():
        return {}
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metadatos sidecar inválidos en {sidecar}: se esperaba un objeto JSON.")
    meta = {"metadata_source": f"sidecar:{sidecar.name}"}
    for key in ("fs_hz", "sps", "symbol_rate_hz", "gain"):
        if key in raw and raw[key] is not None:
            meta[key] = _validate_positive_real(raw[key], f"sidecar.{key}")
    return meta


def _parse_csv_signal(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Archivo CSV vacío o solo comentarios: {path}")

    first_tokens = [tok.strip() for tok in lines[0].split(",")]
    header_keywords = {"i", "q", "real", "imag", "iq", "re", "im"}
    start_idx = 0
    if all(tok.lower() in header_keywords for tok in first_tokens if tok):
        if len(first_tokens) not in (1, 2):
            raise ValueError(f"Cabecera CSV inválida en {path}: {lines[0]}")
        start_idx = 1

    data_lines = lines[start_idx:]
    if not data_lines:
        raise ValueError(f"Archivo CSV sin filas de datos en {path}")

    rows = []
    expected_cols = None
    for idx, line in enumerate(data_lines, start=start_idx + 1):
        parts = [p.strip() for p in line.split(",")]
        if expected_cols is None:
            if len(parts) not in (1, 2):
                raise ValueError(f"Número de columnas inválido ({len(parts)}) en línea {idx} de {path}.")
            expected_cols = len(parts)
        elif len(parts) != expected_cols:
            raise ValueError(f"Inconsistencia de columnas en línea {idx} de {path}.")

        if expected_cols == 2:
            try:
                re_v = float(parts[0])
                im_v = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Valores I/Q no numéricos en línea {idx} de {path}: {line}") from exc
            rows.append(complex(re_v, im_v))
        else:
            token = parts[0].replace(" ", "")
            try:
                if "j" not in token.lower():
                    raise ValueError("Falta parte imaginaria explícita 'j'.")
                rows.append(complex(token))
            except ValueError as exc:
                raise ValueError(f"Complejo inválido en línea {idx} de {path}: {line}") from exc

    samples = np.asarray(rows, dtype=np.complex128)
    if samples.size < 2:
        raise ValueError(f"La señal en {path} tiene menos de dos muestras.")
    if not np.isfinite(samples).all():
        raise ValueError(f"La señal en {path} contiene valores infinitos o NaN.")
    return samples


def load_signal(file_path, dtype="complex64", return_metadata=False):
    """Carga una observación desde formatos documentados (.c64, .bin, .npy, .npz, .csv).

    Rechaza formatos no soportados, binarios truncados, lotes 2D y valores no finitos.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Archivo de señal no encontrado: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{ext}' no soportada. Formatos válidos: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    metadata = _load_sidecar_metadata(path)

    if ext in (".c64", ".bin"):
        try:
            dt = np.dtype(np.complex64 if dtype == "complex64" else dtype)
        except Exception as exc:
            raise ValueError(f"dtype binario inválido: {dtype}") from exc
        if not np.issubdtype(dt, np.complexfloating):
            raise ValueError(f"Los binarios .c64/.bin requieren dtype complejo (recibido: {dt}).")
        file_bytes = path.stat().st_size
        if file_bytes == 0:
            raise ValueError(f"Archivo binario vacío: {path}")
        if file_bytes % dt.itemsize != 0:
            raise ValueError(
                f"Archivo binario truncado o incompleto ({file_bytes} bytes no múltiplo de {dt.itemsize}): {path}"
            )
        samples = np.fromfile(str(path), dtype=dt)
        if samples.size < 2:
            raise ValueError(f"La señal en {path} tiene menos de dos muestras.")
        if not np.isfinite(samples).all():
            raise ValueError(f"La señal en {path} contiene valores infinitos o NaN.")
        samples = np.asarray(samples, dtype=np.complex128)

    elif ext == ".npy":
        try:
            arr = np.load(str(path), allow_pickle=False)
        except ValueError as exc:
            raise ValueError(f"No se pudo cargar .npy seguro sin pickle en {path}: {exc}") from exc
        samples = _extract_iq_vector(arr, str(path))

    elif ext == ".npz":
        try:
            with np.load(str(path), allow_pickle=False) as data:
                candidate_keys = [k for k in data.files if k in ("iq", "iq256", "iq512", "samples")]
                if len(candidate_keys) != 1:
                    raise ValueError(
                        f"El archivo .npz {path} debe contener exactamente una clave de señal "
                        f"entre ('iq', 'iq256', 'iq512', 'samples'); encontradas: {data.files}."
                    )
                samples = _extract_iq_vector(data[candidate_keys[0]], f"{path}:{candidate_keys[0]}")
                for mkey in ("fs_hz", "sps", "symbol_rate_hz", "gain"):
                    if mkey in data and mkey not in metadata:
                        arr_m = np.asarray(data[mkey])
                        if arr_m.size != 1:
                            raise ValueError(f"Metadato '{mkey}' en {path} debe ser escalar.")
                        metadata[mkey] = _validate_positive_real(float(arr_m.reshape(-1)[0]), f"npz.{mkey}")
                        metadata.setdefault("metadata_source", "npz_metadata")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Archivo .npz inválido en {path}: {exc}") from exc

    elif ext == ".csv":
        samples = _parse_csv_signal(path)

    if return_metadata:
        return samples, metadata
    return samples


def analyze(iq, fs_hz=None, detection_calibration=None, noise_reference=None, nfft=None):
    """Integración original de Fase 4 (mantiene retrocompatibilidad)."""
    detection = (
        detect_energy(iq, detection_calibration)
        if detection_calibration is not None
        else {"detected": None, "energy": None, "threshold": None}
    )
    return {
        "detection": detection,
        "spectrum": estimate_spectrum(iq, fs_hz=fs_hz, nfft=nfft),
        "snr": estimate_snr(iq, reference=noise_reference, detected=detection["detected"]),
    }


def _select_eval_windows(iq: np.ndarray, model_window_len: int, active_256_indices: list[int]) -> np.ndarray:
    num_samples = len(iq)
    n_total = num_samples // model_window_len
    if n_total == 0:
        return np.empty((0, model_window_len), dtype=np.complex128)
    all_windows = iq[: n_total * model_window_len].reshape(n_total, model_window_len)
    if not active_256_indices:
        return all_windows
    if model_window_len == 256:
        valid = [i for i in active_256_indices if i < n_total]
        return all_windows[valid] if valid else all_windows
    # Para 512, incluir bloques que solapan al menos una ventana activa de 256
    active_set = set(active_256_indices)
    valid_512 = [k for k in range(n_total) if (2 * k in active_set) or (2 * k + 1 in active_set)]
    return all_windows[valid_512] if valid_512 else all_windows


def _compute_calibrated_confidence(
    mean_margin: float,
    vote_fraction: float,
    n_eval_windows: int,
    model_sha256: str,
    model_window_len: int,
    classes: tuple[str, ...],
    calibrator_path: Path,
    explicit_policy_path: str | Path | None,
) -> tuple[float, bool, str, str]:
    agg_mode = "single_window" if n_eval_windows == 1 else "majority_vote"
    uncal_score = float(1.0 / (1.0 + np.exp(-0.8 * mean_margin)))

    if explicit_policy_path is not None:
        p_path = Path(explicit_policy_path)
        if not p_path.is_file():
            return round(uncal_score, 4), False, "uncalibrated_policy_not_found", "uncalibrated_heuristic_margin_score"
        try:
            pol = joblib.load(p_path)
        except Exception:
            return round(uncal_score, 4), False, "uncalibrated_policy_unreadable", "uncalibrated_heuristic_margin_score"
        if isinstance(pol, dict) and "calibrators" in pol:
            cal_pkg = pol
        else:
            return (
                round(uncal_score, 4),
                False,
                "uncalibrated_legacy_policy_not_validated_for_file_decision",
                "uncalibrated_heuristic_margin_score",
            )
    else:
        if not calibrator_path.is_file():
            return round(uncal_score, 4), False, "uncalibrated_missing_calibrator", "uncalibrated_heuristic_margin_score"
        cal_pkg = joblib.load(calibrator_path)

    if cal_pkg.get("feature_version") != FEATURE_VERSION or tuple(cal_pkg.get("classes", ())) != tuple(classes):
        return round(uncal_score, 4), False, "uncalibrated_incompatible_feature_or_classes", "uncalibrated_heuristic_margin_score"

    matched_entry = None
    for entry in cal_pkg.get("calibrators", {}).values():
        if (
            entry.get("model_sha256") == model_sha256
            and int(entry.get("window_len", -1)) == model_window_len
            and entry.get("aggregation_mode") == agg_mode
            and int(entry.get("expected_n_windows", -1)) == n_eval_windows
        ):
            matched_entry = entry
            break

    if matched_entry is None:
        return (
            round(uncal_score, 4),
            False,
            f"uncalibrated_no_match_for_model_or_{n_eval_windows}_windows",
            "uncalibrated_heuristic_margin_score",
        )

    feat_names = matched_entry.get("feature_names", [])
    if feat_names == ["top2_margin"]:
        feat_vec = [[mean_margin]]
    elif feat_names == ["mean_top2_margin", "vote_fraction"]:
        feat_vec = [[mean_margin, vote_fraction]]
    else:
        return round(uncal_score, 4), False, "uncalibrated_unknown_feature_schema", "uncalibrated_heuristic_margin_score"

    prob = float(matched_entry["calibrator"].predict_proba(feat_vec)[0, 1])
    return (
        round(prob, 4),
        True,
        f"calibrated_{agg_mode}_{n_eval_windows}x{model_window_len}",
        "calibrated_file_decision_correctness_probability",
    )


def analyze_complete(
    iq,
    signal_id="unknown_signal",
    fs_hz=None,
    model_path=None,
    policy_path=None,
    calibrator_path=None,
    detection_calibration=None,
    noise_reference=None,
    default_threshold=None,
    pfa_file=0.01,
    gain=1.0,
    signal_metadata=None,
    plot_path=None,
    classify=True,
):
    """Ejecuta el análisis completo de las cinco salidas obligatorias bajo contrato explícito."""
    warnings = []
    iq_arr = np.asarray(iq, dtype=np.complex128)
    if iq_arr.ndim != 1 or iq_arr.size < 2 or not np.isfinite(iq_arr).all():
        raise ValueError("iq debe ser un vector 1D complejo finito con al menos 2 muestras.")
    num_samples = int(iq_arr.size)
    gain_val = _validate_positive_real(gain, "gain")
    signal_metadata = dict(signal_metadata or {})

    if fs_hz is not None:
        fs_for_spectrum = _validate_positive_real(fs_hz, "fs_hz")
        fs_source = "explicit_argument"
    elif "fs_hz" in signal_metadata and signal_metadata["fs_hz"] is not None:
        fs_for_spectrum = _validate_positive_real(signal_metadata["fs_hz"], "signal_metadata.fs_hz")
        fs_source = str(signal_metadata.get("metadata_source", "file_metadata"))
    else:
        fs_for_spectrum = None
        fs_source = "unspecified"
        if classify:
            warnings.append(
                "Frecuencia de muestreo no especificada; el espectro usa ciclos/muestra y el clasificador queda fuera de alcance."
            )

    # 1. Detección de presencia multi-ventana controlada por archivo
    det_window_len = 256
    det_res = detect_energy_file(
        iq_arr,
        window_len=det_window_len,
        calibration=detection_calibration,
        noise_reference=noise_reference,
        manual_threshold=default_threshold if (detection_calibration is None and noise_reference is None) else None,
        pfa_file=pfa_file,
        fs_hz=fs_for_spectrum,
        gain=gain_val,
    )
    is_detected = det_res["detected"]
    if is_detected is None:
        warnings.append(f"Detección indeterminada ({det_res['calibration_status']}).")
    elif det_res["calibration_status"] == "manual_threshold_unvalidated":
        warnings.append("Detección ejecutada con umbral manual no calibrado sobre referencia de ruido.")

    detection_out = {
        "detected": is_detected,
        "observed_energy": det_res["observed_energy"],
        "mean_energy": det_res["mean_energy"],
        "threshold": det_res["threshold"],
        "threshold_per_window": det_res["threshold_per_window"],
        "threshold_per_file": det_res["threshold_per_file"],
        "normalized_statistic": det_res["normalized_statistic"],
        "active_window_indices": det_res["active_window_indices"],
        "num_windows_evaluated": det_res["num_windows_evaluated"],
        "pfa_file_target": det_res["pfa_file_target"],
        "status": det_res["status"],
        "calibration_status": det_res["calibration_status"],
        "decision_rule": det_res["decision_rule"],
    }

    # 2. Espectro y Ancho de Banda (PSD)
    if is_detected is True and det_res["active_window_indices"] and num_samples >= det_window_len:
        active_iq = np.concatenate(
            [iq_arr[idx * det_window_len : (idx + 1) * det_window_len] for idx in det_res["active_window_indices"]]
        )
        spec_window = active_iq[: min(len(active_iq), 4096)]
    else:
        active_iq = None
        spec_window = iq_arr[: min(num_samples, 4096)]

    noise_pow_for_spec = None
    if is_detected is True and noise_reference is not None and noise_reference.get("method") == "independent_observed_noise_reference":
        noise_pow_for_spec = noise_reference.get("noise_power_estimate")

    spec_res = estimate_spectrum(spec_window, fs_hz=fs_for_spectrum, noise_power_estimate=noise_pow_for_spec)
    bw_val = float(spec_res["total_power_bandwidth"]) if spec_res["total_power_bandwidth"] is not None else None
    bw_unit = "Hz" if fs_for_spectrum is not None else "ciclos/muestra"
    freq_bounds = (
        [float(spec_res["total_power_bounds"][0]), float(spec_res["total_power_bounds"][1])]
        if spec_res["total_power_bounds"] is not None
        else None
    )

    spectrum_out = {
        "bandwidth_99": bw_val,
        "bandwidth_type": "total_power_99_includes_noise",
        "bandwidth_unit": bw_unit,
        "frequency_bounds": freq_bounds,
        "integrated_power": float(spec_res["integrated_power"]),
        "signal_bandwidth_est": spec_res["signal_bandwidth_est"],
        "signal_bandwidth_bounds": spec_res["signal_bandwidth_bounds"],
        "signal_bandwidth_status": (
            spec_res["signal_bandwidth_status"]
            if is_detected is True
            else ("requires_detected_signal" if is_detected is False else "requires_valid_detection")
        ),
        "window_method": spec_res["window"],
        "spectrum_file": str(plot_path) if plot_path else None,
    }

    # 3. Estimación de SNR condicionada
    snr_target_iq = active_iq if active_iq is not None else iq_arr[: min(num_samples, det_window_len)]

    snr_raw = estimate_snr(
        snr_target_iq,
        reference=noise_reference,
        detected=is_detected,
        fs_hz=fs_for_spectrum,
        gain=gain_val,
    )
    if noise_reference is None:
        snr_out = {
            "snr_est_db": None,
            "status": "no_identificable_sin_supuestos",
            "reason": snr_raw["reason"] if is_detected is not True else "falta_referencia_de_piso_de_ruido",
            "method": None,
            "explanation": "La estimación rigurosa de SNR requiere una referencia independiente de ruido en la misma banda y ganancia.",
            "observed_power": float(snr_raw["observed_power"]),
            "noise_power_estimate": None,
            "signal_power_estimate": None,
        }
        warnings.append("SNR indeterminada: se requiere referencia externa de ruido estacionario.")
    else:
        snr_out = {
            "snr_est_db": float(snr_raw["snr_db"]) if snr_raw.get("snr_db") is not None else None,
            "status": snr_raw.get("status", "indeterminada"),
            "reason": snr_raw.get("reason"),
            "method": snr_raw.get("method"),
            "observed_power": float(snr_raw["observed_power"]),
            "noise_power_estimate": snr_raw.get("noise_power_estimate"),
            "signal_power_estimate": snr_raw.get("signal_power_estimate"),
        }

    # 4 y 5. Clasificación y Confianza
    m_path = Path(model_path) if model_path else DEFAULT_SVM_MODEL
    cal_path = Path(calibrator_path) if calibrator_path else DEFAULT_CALIBRATOR
    classification_out = {
        "modulation": None,
        "status": "pending",
        "confidence": None,
        "confidence_calibrated": False,
        "confidence_status": "not_evaluated",
        "calibration_unit": None,
        "model_used": None,
        "model_family": None,
        "model_sha256": None,
        "model_window_len": None,
        "scope": "signal_present_closed_set_6_classes",
        "closed_set_classes": list(CLASSES),
        "window_votes": {},
        "windows_evaluated": 0,
    }

    if not classify:
        classification_out["status"] = "delegated"
    elif is_detected is False:
        classification_out["modulation"] = "NO_SIGNAL"
        classification_out["status"] = "no_signal_detected"
        classification_out["confidence_status"] = "not_applicable_no_signal"
        warnings.append("Detector negativo: modulación marcada como NO_SIGNAL.")
    elif is_detected is None:
        classification_out["modulation"] = "INDETERMINATE_DETECTION"
        classification_out["status"] = det_res["status"]
        classification_out["confidence_status"] = "not_applicable_indeterminate_detection"
        warnings.append("Detección indeterminada: clasificación omitida sin referencia ni umbral validado.")
    elif not m_path.is_file():
        classification_out["status"] = "model_not_found"
        classification_out["confidence_status"] = "model_not_found"
        warnings.append(f"Modelo clasificador no encontrado en {m_path}; clasificación omitida.")
    else:
        bundle = joblib.load(m_path)
        _validate_artifact(bundle)
        pipe = bundle["pipeline"]
        classes = tuple(bundle.get("classes", CLASSES))
        model_window_len = int(pipe.named_steps["features"].n_samples)
        model_sha256 = _sha256_file(m_path)

        classification_out["model_used"] = str(m_path)
        classification_out["model_family"] = str(bundle.get("metadata", {}).get("selected_candidate", {}).get("family", "conventional"))
        classification_out["model_sha256"] = model_sha256
        classification_out["model_window_len"] = model_window_len
        classification_out["closed_set_classes"] = list(classes)

        meta_sps = signal_metadata.get("sps")
        meta_rs = signal_metadata.get("symbol_rate_hz")
        if num_samples < model_window_len:
            classification_out["modulation"] = "OUT_OF_SCOPE_SHORT_SIGNAL"
            classification_out["status"] = "insufficient_samples_for_model"
            classification_out["confidence_status"] = "out_of_scope_short_signal"
            warnings.append(
                f"Señal de {num_samples} muestras menor que la ventana del modelo ({model_window_len}); no se rellena con ceros."
            )
        elif fs_for_spectrum is None:
            classification_out["modulation"] = "OUT_OF_SCOPE_MISSING_FS"
            classification_out["status"] = "out_of_scope_missing_fs_metadata"
            classification_out["confidence_status"] = "out_of_scope_missing_fs"
            warnings.append("No se clasifica sin frecuencia de muestreo verificable compatible con el modelo (8000 Hz).")
        elif (
            not np.isclose(fs_for_spectrum, EXPECTED_MODEL_FS_HZ, rtol=1e-9, atol=0.0)
            or (meta_sps is not None and not np.isclose(float(meta_sps), EXPECTED_MODEL_SPS, rtol=1e-9, atol=0.0))
            or (meta_rs is not None and not np.isclose(float(meta_rs), EXPECTED_MODEL_SYMBOL_RATE_HZ, rtol=1e-9, atol=0.0))
        ):
            classification_out["modulation"] = "OUT_OF_SCOPE_INCOMPATIBLE_SAMPLING"
            classification_out["status"] = "out_of_scope_incompatible_fs_or_sps"
            classification_out["confidence_status"] = "out_of_scope_incompatible_sampling"
            warnings.append(
                f"Muestreo incompatible con el clasificador entrenado (esperado fs={EXPECTED_MODEL_FS_HZ} Hz, sps={EXPECTED_MODEL_SPS})."
            )
        else:
            eval_windows = _select_eval_windows(iq_arr, model_window_len, det_res["active_window_indices"])
            n_eval = int(len(eval_windows))
            preds_idx = pipe.predict(eval_windows).astype(int)
            window_preds = [classes[i] for i in preds_idx]

            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(eval_windows)
                if scores.ndim == 1:
                    scores = scores[None, :]
                sorted_scores = np.sort(scores, axis=1)[:, ::-1]
                margins = sorted_scores[:, 0] - sorted_scores[:, 1]
                mean_margin = float(np.mean(margins))
                summed_scores = np.sum(scores, axis=0)
            else:
                margins = None
                mean_margin = None
                summed_scores = None

            vote_counts = {cls: int(np.sum(preds_idx == idx)) for idx, cls in enumerate(classes) if np.any(preds_idx == idx)}
            max_votes = max(vote_counts.values())
            tied_indices = [idx for idx, cls in enumerate(classes) if vote_counts.get(cls, 0) == max_votes]
            if len(tied_indices) == 1 or summed_scores is None:
                best_idx = tied_indices[0]
            else:
                best_idx = max(tied_indices, key=lambda idx: float(summed_scores[idx]))

            best_class = classes[best_idx]
            vote_fraction = float(max_votes / n_eval)
            classification_out["modulation"] = str(best_class)
            classification_out["status"] = "classified_closed_set"
            classification_out["window_votes"] = vote_counts
            classification_out["windows_evaluated"] = n_eval

            if mean_margin is not None:
                conf_val, conf_cal, conf_status, cal_unit = _compute_calibrated_confidence(
                    mean_margin=mean_margin,
                    vote_fraction=vote_fraction,
                    n_eval_windows=n_eval,
                    model_sha256=model_sha256,
                    model_window_len=model_window_len,
                    classes=classes,
                    calibrator_path=cal_path,
                    explicit_policy_path=policy_path,
                )
                classification_out["confidence"] = conf_val
                classification_out["confidence_calibrated"] = conf_cal
                classification_out["confidence_status"] = conf_status
                classification_out["calibration_unit"] = cal_unit
            else:
                classification_out["confidence"] = round(vote_fraction, 4)
                classification_out["confidence_calibrated"] = False
                classification_out["confidence_status"] = "uncalibrated_vote_fraction_no_margin"
                classification_out["calibration_unit"] = "vote_fraction"

    # 6. Generación del gráfico si fue solicitado
    if plot_path:
        plot_p = Path(plot_path)
        plot_p.parent.mkdir(parents=True, exist_ok=True)
        generate_report_plot(
            iq=iq_arr,
            spec_res=spec_res,
            output_png=plot_p,
            fs_hz=fs_for_spectrum,
            title=f"Reporte de Señal: {signal_id} ({classification_out['modulation']})",
        )

    return {
        "signal_id": signal_id,
        "num_samples": num_samples,
        "fs_hz": fs_for_spectrum,
        "fs_source": fs_source,
        "detection": detection_out,
        "spectrum": spectrum_out,
        "snr": snr_out,
        "classification": classification_out,
        "warnings": warnings,
    }


def generate_report_plot(iq, spec_res, output_png, fs_hz=None, title=None):
    """Genera gráfico de reporte en PNG con densidad espectral (PSD) y constelación."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    freqs = spec_res["frequency"]
    psd_db = 10 * np.log10(np.maximum(spec_res["psd"], 1e-12))
    axes[0].plot(freqs, psd_db, color="royalblue", lw=1.5, label="PSD Observada")

    x_unit = "Hz" if fs_hz is not None else "ciclos/muestra"
    if spec_res["total_power_bounds"] is not None:
        b_min, b_max = spec_res["total_power_bounds"]
        mask = (freqs >= b_min) & (freqs <= b_max)
        axes[0].fill_between(freqs[mask], psd_db[mask], -100, color="lightblue", alpha=0.5, label="Banda 99% (total)")
        axes[0].axvline(b_min, color="crimson", linestyle="--", alpha=0.7, label=f"Límite: {b_min:.3g}")
        axes[0].axvline(b_max, color="crimson", linestyle="--", alpha=0.7, label=f"Límite: {b_max:.3g}")
        bw_str = f"{spec_res['total_power_bandwidth']:.3g}"
    else:
        bw_str = "N/A"

    axes[0].set_xlabel(f"Frecuencia ({x_unit})")
    axes[0].set_ylabel(f"PSD (dB/{'Hz' if fs_hz is not None else 'ciclo'})")
    axes[0].set_title(f"Densidad Espectral (BW 99% total = {bw_str} {x_unit})")
    axes[0].set_ylim([max(-90, float(psd_db.min()) - 5), max(10, float(psd_db.max()) + 5)])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8)

    sub_iq = iq[: min(len(iq), 1024)]
    axes[1].scatter(np.real(sub_iq), np.imag(sub_iq), alpha=0.5, color="darkorange", s=14, edgecolors="none")
    axes[1].set_xlabel("En-Fase (I)")
    axes[1].set_ylabel("Cuadratura (Q)")
    axes[1].set_title(f"Diagrama de Constelación ({len(sub_iq)} muestras)")
    max_val = max(1.5, float(np.max(np.abs(sub_iq))) * 1.1)
    axes[1].set_xlim([-max_val, max_val])
    axes[1].set_ylim([-max_val, max_val])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].axhline(0, color="gray", lw=0.8, alpha=0.5)
    axes[1].axvline(0, color="gray", lw=0.8, alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_png), dpi=160)
    plt.close()


def main():
    """Punto de entrada CLI para el monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor Integrado de Cinco Salidas para Señales Desconocidas (COM2)"
    )
    parser.add_argument("--input", "-i", required=True, help="Ruta al archivo de señal (.c64, .bin, .npy, .npz, .csv)")
    parser.add_argument("--fs-hz", type=float, default=None, help="Frecuencia de muestreo en Hz (opcional; si falta se usa ciclos/muestra)")
    parser.add_argument("--dtype", default="complex64", help="Tipo de dato binario (por defecto complex64)")
    parser.add_argument("--model", "-m", default=str(DEFAULT_SVM_MODEL), help="Ruta al modelo clasificador .joblib")
    parser.add_argument("--calibrator", default=str(DEFAULT_CALIBRATOR), help="Ruta al paquete de calibración validado .joblib")
    parser.add_argument("--policy", default=None, help="Ruta opcional a política externa (no validada salvo esquema compatible)")
    parser.add_argument("--output", "-o", default=None, help="Ruta al archivo JSON de resultados")
    parser.add_argument("--plot", "-p", default=None, help="Ruta a la figura PNG de espectro y constelación")
    parser.add_argument("--noise-ref", default=None, help="Ruta a archivo de ruido de referencia para calibrar detección y SNR")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral manual opcional de energía (sin calibración estadística)")
    parser.add_argument("--gain", type=float, default=1.0, help="Ganancia declarada de captura para verificar compatibilidad con referencia")
    parser.add_argument("--motor", choices=("svm", "cascada"), default="svm", help="Clasificador: SVM original o cascada independiente del generador")
    parser.add_argument("--paquete", default=None, help="Paquete de modelos de la cascada")
    parser.add_argument("--sps", type=float, default=None, help="Muestras por símbolo, si se conocen (solo cascada)")

    args = parser.parse_args()

    try:
        input_path = Path(args.input)
        signal_id = input_path.stem

        results_dir = Path("results/monitor")
        out_json = Path(args.output) if args.output else results_dir / f"{signal_id}.json"
        out_plot = Path(args.plot) if args.plot else results_dir / f"{signal_id}_psd.png"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        iq, sig_meta = load_signal(input_path, dtype=args.dtype, return_metadata=True)

        effective_fs = args.fs_hz if args.fs_hz is not None else sig_meta.get("fs_hz")
        if effective_fs is not None:
            effective_fs = _validate_positive_real(effective_fs, "fs_hz")
        gain_val = _validate_positive_real(args.gain, "gain")

        noise_ref = None
        if args.noise_ref:
            noise_samples, noise_meta = load_signal(args.noise_ref, dtype=args.dtype, return_metadata=True)
            noise_fs = noise_meta.get("fs_hz", effective_fs)
            noise_gain = noise_meta.get("gain", gain_val)
            noise_ref = calibrate_noise_reference(noise_samples, fs_hz=noise_fs, gain=noise_gain)

        res = analyze_complete(
            iq=iq,
            signal_id=signal_id,
            fs_hz=args.fs_hz,
            model_path=args.model,
            policy_path=args.policy,
            calibrator_path=args.calibrator,
            noise_reference=noise_ref,
            default_threshold=args.threshold,
            gain=gain_val,
            signal_metadata=sig_meta,
            plot_path=out_plot,
            classify=args.motor == "svm",
        )
        if args.motor == "cascada":
            from src.cascada.pipeline import PAQUETE_DEFECTO, cargar_paquete, clasificacion_monitor

            paquete = cargar_paquete(args.paquete or PAQUETE_DEFECTO)
            res["classification"] = clasificacion_monitor(iq, res["detection"]["detected"], paquete, args.sps)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    det = res["detection"]
    det_str = (
        "SÍ (Señal presente)"
        if det["detected"] is True
        else ("NO (Solo ruido / no detectada)" if det["detected"] is False else f"INDETERMINADA ({det['status']})")
    )
    thresh_str = f"{det['threshold']:.4g}" if det["threshold"] is not None else "N/A"
    fs_str = f"{res['fs_hz']} Hz ({res['fs_source']})" if res["fs_hz"] is not None else "No especificada (ciclos/muestra)"
    bw_str = f"{res['spectrum']['bandwidth_99']:.4g}" if res["spectrum"]["bandwidth_99"] is not None else "N/A"
    bounds = res["spectrum"]["frequency_bounds"]
    bounds_str = f"[{bounds[0]:.4g}, {bounds[1]:.4g}]" if bounds is not None else "N/A"
    snr_val = res["snr"]["snr_est_db"]
    snr_str = f"{snr_val:.2f} dB" if snr_val is not None else "null"
    conf_val = res["classification"]["confidence"]
    conf_str = f"{conf_val * 100:.1f}%" if conf_val is not None else "null"

    print("\n" + "=" * 60)
    print(f" MONITOR INTEGRADO COM2 - REPORTE DE SEÑAL [{signal_id}]")
    print("=" * 60)
    print(f"  Muestras totales   : {res['num_samples']}")
    print(f"  Frecuencia muestreo: {fs_str}")
    print(f"  1. Detección       : {det_str}")
    print(f"     Energía máx/umbral: {det['observed_energy']:.4g} (Umbral: {thresh_str})")
    print(f"  2. Ancho de Banda  : {bw_str} {res['spectrum']['bandwidth_unit']} (99% potencia total)")
    print(f"     Banda ocupada   : {bounds_str}")
    print(f"  3. SNR Estimada    : {snr_str} ({res['snr']['status']})")
    print(f"  4. Modulación      : {res['classification']['modulation']} ({res['classification']['status']})")
    print(f"  5. Confianza       : {conf_str} (Calibrada: {res['classification']['confidence_calibrated']})")
    print("-" * 60)
    print(f"  Archivo JSON       : {out_json}")
    print(f"  Gráfico PSD / Const: {out_plot}")
    if res["warnings"]:
        print("  Advertencias:")
        for w in res["warnings"]:
            print(f"    * {w}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
