from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    service_name: str = Field(default="alpha-bank-anti-deepfake-ml", alias="ML_SERVICE_NAME")
    risk_api_url: str | None = Field(default=None, alias="RISK_API_URL")
    enable_risk_api_callbacks: bool = Field(default=False, alias="ML_ENABLE_RISK_API_CALLBACKS")
    callback_timeout_seconds: float = Field(default=2.0, alias="ML_CALLBACK_TIMEOUT_SECONDS")
    cors_origins: str = Field(default="*", alias="ML_CORS_ORIGINS")

    decision_allow_threshold: float = Field(default=0.35, alias="ML_DECISION_ALLOW_THRESHOLD")
    decision_deny_threshold: float = Field(default=0.72, alias="ML_DECISION_DENY_THRESHOLD")

    challenge_ttl_seconds: int = Field(default=900, alias="ML_CHALLENGE_TTL_SECONDS")
    active_light_min_correlation: float = Field(default=0.55, alias="ML_ACTIVE_LIGHT_MIN_CORRELATION")
    active_light_min_contrast: float = Field(default=0.08, alias="ML_ACTIVE_LIGHT_MIN_CONTRAST")
    active_light_min_temporal_correlation: float = Field(default=0.65, alias="ML_ACTIVE_LIGHT_MIN_TEMPORAL_CORRELATION")
    active_light_min_pair_count: int = Field(default=4, alias="ML_ACTIVE_LIGHT_MIN_PAIR_COUNT")
    active_light_min_spatial_contrast: float = Field(default=0.025, alias="ML_ACTIVE_LIGHT_MIN_SPATIAL_CONTRAST")
    active_light_min_response_snr: float = Field(default=0.04, alias="ML_ACTIVE_LIGHT_MIN_RESPONSE_SNR")
    active_light_min_color_cosine: float = Field(default=0.15, alias="ML_ACTIVE_LIGHT_MIN_COLOR_COSINE")
    rppg_min_signal_quality: float = Field(default=0.35, alias="ML_RPPG_MIN_SIGNAL_QUALITY")

    audio_model_path: str = Field(default="models/audio/wavlm_all4_best.pt", alias="ML_AUDIO_MODEL_PATH")
    asr_model_path: str = Field(default="models/asr/faster-whisper-medium", alias="ML_ASR_MODEL_PATH")
    asr_device: str = Field(default="cpu", alias="ML_ASR_DEVICE")
    asr_compute_type: str = Field(default="int8", alias="ML_ASR_COMPUTE_TYPE")
    asr_cpu_threads: int = Field(default=4, ge=1, alias="ML_ASR_CPU_THREADS")
    video_clip_checkpoint_path: str = Field(
        default="models/video/clip_vit_b16_deepfake_best.pt",
        alias="ML_VIDEO_CLIP_CHECKPOINT_PATH",
    )
    video_convnext_checkpoint_path: str | None = Field(default=None, alias="ML_VIDEO_CONVNEXT_CHECKPOINT_PATH")
    video_xgb_models_dir: str = Field(default="models/xgb", alias="ML_VIDEO_XGB_MODELS_DIR")
    video_xgb_threshold: float = Field(default=0.45, ge=0.0, le=1.0, alias="ML_VIDEO_XGB_THRESHOLD")
    video_device: str = Field(default="auto", alias="ML_VIDEO_DEVICE")
    video_max_inferences: int = Field(default=12, alias="ML_VIDEO_MAX_INFERENCES")
    video_infer_every: int = Field(default=5, alias="ML_VIDEO_INFER_EVERY")


settings = Settings()
