from django.urls import path, include

urlpatterns = [
    path('api/', include('annotation.api.urls'), name='annotation_v1_urls'),
]