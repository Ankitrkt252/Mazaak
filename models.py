from django.db import models
from django_mysql.models import JSONField

from panel.models import ModelConnection


class BaseModel(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Label(BaseModel):
    name = models.CharField(max_length=50)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return f"{self.name}"


class Frame(BaseModel):
    image = models.ImageField(upload_to="annotation/")
    annotated = models.BooleanField(default=False)
    dataset = models.ForeignKey('AnnotationDataset', on_delete=models.CASCADE)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return f"{self.image.name}"


class Classifier(BaseModel):
    ANNOTATION_BY = (
        ('p', 'PERSON'),
        ('m', 'MODEL'),
    )
    annotated_by = models.CharField(max_length=5, choices=ANNOTATION_BY, default='p')
    frame_obj = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name="classifier_data")
    label_class = models.PositiveSmallIntegerField()


class Detector(BaseModel):
    ANNOTATION_BY = (
        ('p', 'PERSON'),
        ('m', 'MODEL'),
    )
    x = models.DecimalField(max_digits=50, decimal_places=20)
    y = models.DecimalField(max_digits=50, decimal_places=20)
    w = models.DecimalField(max_digits=50, decimal_places=20)
    h = models.DecimalField(max_digits=50, decimal_places=20)
    annotated_by = models.CharField(max_length=5, choices=ANNOTATION_BY, default='p')
    frame_obj = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name="detector_data")
    label_class = models.PositiveSmallIntegerField()

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return f"X: {self.x}, Y: {self.y}, W: {self.w}, H: {self.h}"


class Segmentation(BaseModel):
    ANNOTATION_BY = (
        ('p', 'PERSON'),
        ('m', 'MODEL'),
    )
    annotated_by = models.CharField(max_length=5, choices=ANNOTATION_BY, default='p')
    frame_obj = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name="segmentation_data")
    label_class = models.PositiveSmallIntegerField()


class Point(BaseModel):
    segment = models.ForeignKey(Segmentation, on_delete=models.CASCADE, related_name="segmentation_coords")
    x = models.DecimalField(max_digits=50, decimal_places=20)
    y = models.DecimalField(max_digits=50, decimal_places=20)


class AnnotationDataset(BaseModel):
    MODEL_TYPE = (
        ('d', 'DETECTOR'),
        ('c', 'CLASSIFIER'),
        ('s', 'SEGMENTATION'),
    )

    name = models.CharField(max_length=100)
    version = models.CharField(max_length=50, default="latest")
    description = models.TextField(null=True, blank=True)
    model_type = models.CharField(max_length=5, choices=MODEL_TYPE)
    config = JSONField(default=dict)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return f"{self.name}:{self.version}"


class DeploymentHistory(BaseModel):
    DEPLOYMENET_STATUS = (
        ("training", "training"),
        ("trained", "trained"),
        ("failed", "failed")
    )
    name = models.CharField(max_length=500)
    model = models.ForeignKey(ModelConnection, on_delete=models.SET_NULL, null=True)
    dataset = models.ForeignKey(AnnotationDataset, on_delete=models.SET_NULL, null=True)
    total_data = models.IntegerField(default=0)
    batch_size = models.PositiveSmallIntegerField()
    epochs = models.PositiveSmallIntegerField()
    img_size = models.PositiveSmallIntegerField()
    meta_data = models.TextField()
    status = models.CharField(max_length=30, choices=DEPLOYMENET_STATUS, default=DEPLOYMENET_STATUS[2][0])
    is_active = models.BooleanField(default=False)
    is_deployed = models.BooleanField(default=False)


    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return f"{self.name}"

