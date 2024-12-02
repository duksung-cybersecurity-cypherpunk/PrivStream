from django.apps import AppConfig

class CamConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "webcam"

    def ready(self):
        from yolo_loader import load_global_model
        # 서버가 시작될 때 YOLO 모델을 한 번만 로드
        load_global_model()