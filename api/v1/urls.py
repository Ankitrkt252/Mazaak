from django.urls import path, include
from rest_framework.routers import SimpleRouter


from .views import (
    DetectorViewSet,
    LabelViewSet,
    FrameViewSet,
    DeploymentHistoryViewSet,
    AnnotationDatasetViewSet,
    ModelConnectionViewSet,

    # UploadFromZip,
)

router = SimpleRouter()
router.register(r'detector', DetectorViewSet, basename='annotation_bbox')
router.register(r'label', LabelViewSet, basename='annotation_label')
router.register(r'frame', FrameViewSet, basename='annotation_frame')
router.register(r'dataset', AnnotationDatasetViewSet, basename='annotation_dataset')
router.register(r'deployment', DeploymentHistoryViewSet, basename='annotation_dataset')
router.register(r'models', ModelConnectionViewSet, basename='annotation_dataset')

urlpatterns = [
    path('', include(router.urls)),
]
