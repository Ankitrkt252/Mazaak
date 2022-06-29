from rest_framework import serializers

from panel.models import (
    ModelConnection,
)

from annotation.models import (
    Label,
    Frame,
    Detector,
    Classifier,
    Segmentation,
    Point,
    AnnotationDataset,
    DeploymentHistory,
)


class DetectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detector
        fields = "__all__"
        read_only_fields = ("created", "updated")

    def to_representation(self, instance):
        #self.fields['label_class'] = LabelSerializer(read_only=True)
        return super(self.__class__, self).to_representation(instance)

class ClassifierSerializer(serializers.ModelSerializer):
    class Meta:
        model = Classifier
        fields = "__all__"
        read_only_fields = ("created", "updated")

class ClassifierSerializer(serializers.ModelSerializer):
    class Meta:
        model = Classifier
        fields = "__all__"
        read_only_fields = ("created", "updated")


class PointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Point
        fields = "__all__"
        read_only_fields = ("created", "updated")

class SegmentationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Segmentation
        fields = "__all__"
        read_only_fields = ("created", "updated")

    def to_representation(self, instance):
        self.fields['segmentation_coords'] = PointSerializer(read_only=True, many=True)
        return super(self.__class__, self).to_representation(instance)


class LabelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Label
        fields = "__all__"
        read_only_fields = ("created", "updated")


class FrameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Frame
        fields = "__all__"
        read_only_fields = ("created", "updated")

    def to_representation(self, instance):
        self.fields['dataset'] = AnnotationDatasetSerializer(read_only=True)
        self.fields['detector_data'] = DetectorSerializer(read_only=True, many=True)
        self.fields['classifier_data'] = ClassifierSerializer(read_only=True, many=True)
        self.fields['segmentation_data'] = SegmentationSerializer(read_only=True, many=True)
        # self.fields['annotated_frame'] = AnnotatedFrameSerializer(read_only=True, many=True)
        # self.fields['image'] = FrameSerializer(read_only=True)
        return super(self.__class__, self).to_representation(instance)


class AnnotationDatasetSerializer(serializers.ModelSerializer):
    status = serializers.SerializerMethodField()

    def get_status(self, obj):
        return {
            "total_images": obj.frame_set.filter().count(),
            "annotated_images": obj.frame_set.filter(annotated=True).count(),
        }

    class Meta:
        model = AnnotationDataset
        fields = "__all__"
        read_only_fields = ("created", "updated")

    def to_representation(self, instance):
        self.fields['labels'] = LabelSerializer(read_only=True, many=True)
        return super(self.__class__, self).to_representation(instance)


class DeploymentHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = DeploymentHistory
        fields = "__all__"
        read_only_fields = ("created", "updated")

    def validate(self, validated_data):
        if validated_data.get("dataset"):
            validated_data["total_data"] = validated_data.get("dataset").frame_set.filter(annotated=True).count()
        if validated_data.get("is_deployed"):
            if self.instance:
                DeploymentHistory.objects.filter(is_deployed=True, model=self.instance.model).update(is_deployed=False)
        return super().validate(validated_data)

    def to_representation(self, instance):
        self.fields['model'] = ModelConnectionSerializer(read_only=True)
        self.fields['dataset'] = AnnotationDatasetSerializer(read_only=True)
        return super(self.__class__, self).to_representation(instance)


class ModelConnectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelConnection
        fields = "__all__"

