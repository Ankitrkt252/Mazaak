from django.urls import path, include

urlpatterns = [
    path('v1/', include('annotation.api.v1.urls'), name='annotation_v1_urls'),
]