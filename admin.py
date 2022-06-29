from django.contrib import admin

from annotation.models import (
    Detector,
    Label,
    Frame,
    AnnotationDataset,
)

class DetectorInline(admin.TabularInline):
    model = Detector


@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    pass


@admin.register(Detector)
class DetectorAdmin(admin.ModelAdmin):
    list_display = ('id', 'x', 'y', 'w', 'h', 'created', 'updated')


@admin.register(AnnotationDataset)
class AnnotationDatasetAdmin(admin.ModelAdmin):
    pass



admin.site.register(Label)
