import io
import os
import csv
import uuid
import zipfile
from io import BytesIO

from django.conf import settings
from django.core.files import File
from django.http import HttpResponse
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from rest_framework.views import APIView

import numpy as np
from PIL import Image

from panel.models import (
    ModelConnection,
)

from annotation.models import(
    Detector,
    Classifier,
    Segmentation,
    Point,
    Label,
    Frame,
    BaseModel,
    AnnotationDataset,
    DeploymentHistory,
)


from .serializers import (
    LabelSerializer,
    FrameSerializer,
    DetectorSerializer,
    DeploymentHistorySerializer,
    AnnotationDatasetSerializer,
    ModelConnectionSerializer,
)

# from api.v1.utils.viewsets import ActivityViewset
# from panel.helpers.permission_helper import GenericObjectPermissions


class DetectorViewSet(ModelViewSet):
    serializer_class = DetectorSerializer
    # permission_classes = (GenericObjectPermissions,)
    queryset = Detector.objects.all()
    perms_map = {
        'GET': ['annotation.view_detector'],
        'POST': ['annotation.add_detector'],
        'PATCH': ['annotation.change_detector'],
        'PUT': ['annotation.change_detector'],
        'DELETE': ['annotation.delete_detector'],
    }



class LabelViewSet(ModelViewSet):
    serializer_class = LabelSerializer
    # permission_classes = (GenericObjectPermissions,)
    queryset = Label.objects.all()
    perms_map = {
        'GET': ['annotation.view_label'],
        'POST': ['annotation.add_label'],
        'PATCH': ['annotation.change_label'],
        'PUT': ['annotation.change_label'],
        'DELETE': ['annotation.delete_label'],
    }



class FrameViewSet(ModelViewSet):
    serializer_class = FrameSerializer
    # permission_classes = (GenericObjectPermissions,)
    perms_map = {
        'GET': ['annotation.view_frame'],
        'POST': ['annotation.add_frame'],
        'PATCH': ['annotation.change_frame'],
        'PUT': ['annotation.change_frame'],
        'DELETE': ['annotation.delete_frame'],
    }

    def paginate_queryset(self, queryset):
        if self.request.query_params.get('no_page'):
            return None
        return super().paginate_queryset(queryset)

    def get_queryset(self):
        dataset_id = self.request.GET.get("dataset")
        queryset = Frame.objects.all()
        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        return queryset

    @action(
        methods=['PATCH'], detail=True
    )
    def update_frame(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.dataset.model_type == "d":
            instance.detector_data.filter().delete()
            for data in self.request.data.get("detector_data"):
                Detector.objects.create(**data, frame_obj=instance)
        elif instance.dataset.model_type == "c":
            instance.classifier_data.filter().delete()
            for label in self.request.data.get("classifier_data"):
                Classifier.objects.create(label_class=label, frame_obj=instance)            
        elif instance.dataset.model_type == "s":
            instance.segmentation_data.filter().delete()
            for data in self.request.data.get("segmentation_data"):
                segment_obj = Segmentation.objects.create(label_class=data["label_class"], frame_obj=instance)
                for point in data["points"]:
                    Point.objects.create(segment=segment_obj, x=float(point["x"]), y=float(point["y"]))
        return Response(self.serializer_class(instance).data)


    @action(
        methods=['POST'], detail=False
    )
    def update_frames(self, request, *args, **kwargs):
        frame_list = []
        frames = self.request.data.get("frames", [])
        for data in frames:
            instance = Frame.objects.get(pk=data["frame"])
            frame_list.append(data["frame"])
            if instance.dataset.model_type == "d":
                instance.detector_data.filter().delete()
                for d_data in data["detector_data"]:
                    Detector.objects.create(**d_data, frame_obj=instance)
            elif instance.dataset.model_type == "c":
                instance.classifier_data.filter().delete()
                for label in data["classifier_data"]:
                    Classifier.objects.create(label_class=label, frame_obj=instance)
            elif instance.dataset.model_type == "s":
                instance.segmentation_data.filter().delete()
                for s_data in data["segmentation_data"]:
                    if len(s_data["points"]) < 4:
                        continue
                    segment_obj = Segmentation.objects.create(label_class=s_data["label_class"], frame_obj=instance)
                    for point in s_data["points"]:
                        Point.objects.create(segment=segment_obj, x=float(point["x"]), y=float(point["y"]))

            if instance.detector_data.exists() or instance.classifier_data.exists() or instance.segmentation_data.exists():
                instance.annotated = True
            else:
                instance.annotated = False
            instance.save()

        return Response(self.serializer_class(Frame.objects.filter(id__in=frame_list), many=True).data)



class AnnotationDatasetViewSet(ModelViewSet):
    serializer_class = AnnotationDatasetSerializer
    # permission_classes = (GenericObjectPermissions,)
    queryset = AnnotationDataset.objects.all()
    perms_map = {
        'GET': ['annotation.view_annotationdataset'],
        'POST': ['annotation.add_annotationdataset'],
        'PATCH': ['annotation.change_annotationdataset'],
        'PUT': ['annotation.change_annotationdataset'],
        'DELETE': ['annotation.delete_annotationdataset'],
    }

    def paginate_queryset(self, queryset):
        if self.request.query_params.get('no_page'):
            return None
        return super().paginate_queryset(queryset)


    @action(
        methods=['POST'], detail=False
    )
    def upload_zip(self, request, *args, **kwargs):
        dataset = request.FILES.get("dataset")
        model_type = request.data.get("model_type")

        if not zipfile.is_zipfile(dataset):
            return Response({"Error": "Zip Archieve required"})

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        dataset_kwargs = dict(serializer.validated_data)
        dataset_obj, created = AnnotationDataset.objects.get_or_create(**dataset_kwargs)

        with zipfile.ZipFile(dataset, 'r') as zipObj:
            data_failed_to_save = [ ]
            for fileName in zipObj.namelist():
                if fileName.endswith('.jpg'):
                    data = zipObj.read(fileName)
                    try:
                        image = Image.open(BytesIO(data))
                        image.load()
                        image = Image.open(BytesIO(data))
                        image.verify()
                    except:
                        data_failed_to_save.append(fileName)
                        continue

                    name = str(uuid.uuid4().hex) + ".jpg"
                    path = os.path.join("annotation", str(name))
                    saved_path = default_storage.save(path, ContentFile(data))

                    image = Frame.objects.create(image=saved_path, dataset=dataset_obj)

                    if fileName.replace('.jpg', '.txt') in zipObj.namelist():
                        with zipObj.open(fileName.replace('.jpg', '.txt')) as meta_file:
                            for line in meta_file:
                                if model_type == "d":
                                    label_id, x, y, w, h = line.decode('utf-8').strip().split(" ")
                                    Detector.objects.create(
                                        x=float(x),
                                        y=float(y),
                                        w=float(w),
                                        h=float(h),
                                        frame_obj=image,
                                        annotated_by='p',
                                        label_class=int(label_id),
                                    )
                                elif model_type == "c":
                                    label_id = int(line.decode('utf-8').strip())
                                    Classifier.objects.create(frame_obj=image, label_class=label_id)
                                elif model_type == "s":
                                    label_id, *coords = list(map(float, map(str.strip, line.decode("utf-8").strip().split(" "))))
                                    if len(coords) < 5:
                                        continue
                                    segment_obj = Segmentation.objects.create(frame_obj=image, label_class=int(label_id))
                                    for i in range(0, len(coords), 2):
                                        Point.objects.create(x=coords[i], y=coords[i+1], segment=segment_obj)
                    if image.detector_data.exists() or image.classifier_data.exists() or image.segmentation_data.exists():
                        image.annotated = True
                        image.save()

        return Response({
            "added_in_existed_dataset": not created,
            "new_dataset_created": created,
            "data_failed_to_save": data_failed_to_save,
        })


    @action(
        methods=['GET'], detail=True
    )
    def download(self, request, *args, **kwargs):
        dataset_obj = AnnotationDataset.objects.get(pk=kwargs.get("pk"))

        response = HttpResponse(content_type='application/zip')
        zip_subdir = f"{dataset_obj.name}:{dataset_obj.version}/"
        zip_filename = f"{dataset_obj.name}:{dataset_obj.version}.zip"

        train_frames = int(request.GET.get("train", 0))
        val_frames = int(request.GET.get("val", 0))
        test_frames = int(request.GET.get("test", 0))

        if train_frames or val_frames or test_frames:
            total_frames = Frame.objects.filter(dataset_id=kwargs.get("pk"), annotated=True).count()
            train_frames = train_frames * total_frames // 100
            val_frames = val_frames * total_frames // 100
            test_frames = test_frames * total_frames / 100

        if dataset_obj.model_type == "d":
            with zipfile.ZipFile(response, 'w') as zipObj:
                for frame in Frame.objects.filter(dataset_id=kwargs.get("pk")).order_by("?"):
                    zip_path = zip_subdir
                    labels_zip_path = zip_subdir
                    if train_frames > 0:
                        zip_path += "images/train/"
                        labels_zip_path += "labels/train/"
                        train_frames -= 1
                    elif val_frames > 0:
                        zip_path += "images/val/"
                        labels_zip_path += "labels/val/"
                        val_frames -= 1
                    elif test_frames > 0:
                        zip_path += "images/test/"
                        labels_zip_path += "labels/test/"
                        test_frames -= 1
                    zipObj.write(frame.image.path, zip_path+frame.image.name.strip().split("/")[-1])
                    meta_path = labels_zip_path+frame.image.name.strip().split("/")[-1].replace(".jpg", '.txt')
                    meta_data = ""
                    for label, x, y, w, h in frame.detector_data.values_list("label_class", "x", "y", "w", "h"):
                        meta_data += f'{label} {x} {y} {w} {h}\n'
                    if meta_data:
                        zipObj.writestr(meta_path, bytes(meta_data.encode('utf-8')))

        elif dataset_obj.model_type == "s":
            with zipfile.ZipFile(response, 'w') as zipObj:
                for frame in Frame.objects.filter(dataset_id=kwargs.get("pk")).order_by("?"):
                    zip_path = zip_subdir
                    labels_zip_path = zip_subdir
                    if train_frames > 0:
                        zip_path += "images/train/"
                        labels_zip_path += "labels/train/"
                        train_frames -= 1
                    elif val_frames > 0:
                        zip_path += "images/val/"
                        labels_zip_path += "labels/val/"
                        val_frames -= 1
                    elif test_frames > 0:
                        zip_path += "images/test/"
                        labels_zip_path += "labels/test/"
                        test_frames -= 1
                    zipObj.write(frame.image.path, zip_path+frame.image.name.strip().split("/")[-1])
                    meta_path = labels_zip_path+frame.image.name.strip().split("/")[-1].replace(".jpg", '.txt')
                    meta_data = " "
                    label = None
                    for label, x, y in frame.segmentation_data.prefetch_related("segmentation_coords").values_list(
                        "label_class", "segmentation_coords__x", "segmentation_coords__y"
                    ):
                        meta_data += f'{x} {y}'
                    if label:
                        meta_data = label + meta_data + "\n"
                        zipObj.writestr(meta_path, bytes(meta_data.encode('utf-8')))

        if dataset_obj.model_type == "c":
            main_buffer, train_buffer, val_buffer, test_buffer = None, None, None, None
            main_writer, train_writer, val_writer, test_writer = None, None, None, None

            total_frames = Frame.objects.filter(dataset_id=kwargs.get("pk")).count()
            if train_frames:
                train_frames = train_frames * total_frames / 100
                train_buffer = io.StringIO()
                train_writer = csv.writer(train_buffer, delimiter='\t')

            if val_frames:
                val_frames = val_frames * total_frames / 100
                val_buffer = io.StringIO()
                val_writer = csv.writer(val_buffer, delimiter='\t')

            if test_frames:
                test_frames = test_frames * total_frames / 100
                test_buffer = io.StringIO()
                test_writer = csv.writer(test_buffer, delimiter='\t')

            main_buffer = io.StringIO()
            main_writer = csv.writer(main_buffer, delimiter='\t')


            with zipfile.ZipFile(response, 'w') as zipObj:
                for frame in Frame.objects.filter(dataset_id=kwargs.get("pk")).order_by("?"):
                    writer = main_writer
                    zip_path = zip_subdir
                    labels_zip_path = zip_subdir
                    if train_frames > 0:
                        zip_path += "images/train/"
                        labels_zip_path += "labels/train/"
                        writer = train_writer
                        train_frames -= 1
                    elif val_frames > 0:
                        zip_path += "images/val/"
                        labels_zip_path += "labels/val/"
                        writer = val_writer
                        val_frames -= 1
                    elif test_frames > 0:
                        zip_path += "images/test/"
                        labels_zip_path += "labels/test/"
                        writer = test_writer
                        test_frames -= 1
                    zipObj.write(frame.image.path, zip_path+frame.image.name.strip().split("/")[-1])
                    for label in frame.classifier_data.values_list("label_class", flat=True):
                        writer.writerow([(zip_path+frame.image.name.strip().split("/")[-1]).encode('utf-8'), str(label).encode('utf-8')])

                if main_writer:
                    buffer = main_buffer.getvalue()
                    zipObj.writestr(zip_subdir + "main.csv", bytes(buffer.encode('utf-8')))
                if train_writer:
                    buffer = train_buffer.getvalue()
                    zipObj.writestr(zip_subdir + "train.csv", bytes(buffer.encode('utf-8')))
                if val_writer:
                    buffer = val_buffer.getvalue()
                    zipObj.writestr(zip_subdir + "val.csv", bytes(buffer.encode('utf-8')))
                if test_writer:
                    buffer = test_buffer.getvalue()
                    zipObj.writestr(zip_subdir + "test.csv", bytes(buffer.encode('utf-8')))

        response['Content-Disposition'] = 'attachment; filename=%s' % zip_filename
        return response


    def _create_annotation(self, captured_dataset, dataset_obj, model_type, data_name=None):
        """
            {
                bboxs: []
                polys: [(x, y), (x, y), (x, y)]
                labels: [0, 1, 2, 3, ]
                image_path: "/media/{uuid.uuid4().hex}.jpg"
            }
        """


    def create_annotation(self, captured_dataset, dataset_obj, model_type, data_name=None):
        if data_name:
            self.create_annotation(captured_dataset, dataset_obj, model_type, data_name)
            del captured_dataset[data_name]
        else:
            for data_name in list(captured_dataset.keys()):
                self.create_annotation(captured_dataset, dataset_obj, model_type, data_name)
                del captured_dataset[data_name]

    @action(
        methods=['POST'], detail=False
    )

    def upload(self, request, *args, **kwargs):
        dataset = request.FILES.getlist("dataset")
        model_type = request.data.get("model_type")

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        dataset_obj = serializer.save()

        labels = [ ]
        annotated_frames = [ ]
        data_failed_to_save = [ ]
        captured_dataset = { }
        for data in dataset:
            if data.name.endswith('.jpg'):
                try:
                    image = Image.open(data)
                    image.load()
                    image = Image.open(data)
                    image.verify()
                except:
                    data_failed_to_save.append(data.name)
                    continue
                name = str(uuid.uuid4().hex) + ".jpg"
                path = os.path.join(settings.MEDIA_ROOT, "annotation", str(name))
                saved_path = default_storage.save(path, data)
                image = Frame.objects.create(image=saved_path)
                if data.name in captured_dataset:
                    captured_dataset[data.name]["image_path"] = saved_path
                    self.create_annotation(captured_dataset, dataset_obj, model_type, data.name)
                else:
                    captured_dataset[data.name.replace('.jpg', '.txt')] = dict(
                        image_path=saved_path
                    )
            elif data.name.endswith('.txt'):
                with open(data.name.replace('.jpg', '.txt')) as meta_file:
                    bboxs = [ ]
                    polys = [ ]
                    bboxs_labels = [ ]
                    for line in meta_file:
                        if model_type == "d":
                            label_id, x, y, w, h = line.decode('utf-8').strip().split(" ")
                            bboxs.append(dict(x=float(x), y=float(y), w=float(w), h=float(h)))
                            labels.append(int(label_id)+1)
                            bboxs_labels.append(int(label_id)+1)
                        elif model_type == "c":
                            label_id = int(line.decode('utf-8').strip())
                            labels.append(int(label_id)+1)
                            bboxs_labels.append(int(label_id)+1)
                        elif model_type == "s":
                            pass

                    if data.name in captured_dataset:
                        captured_dataset[data.name]["bboxs"] = bboxs
                        captured_dataset[data.name]["polys"] = polys
                        captured_dataset[data.name]["labels"] = bboxs_labels
                        self.create_annotation(captured_dataset, dataset_obj, model_type, data.name)
                    else:
                        captured_dataset[data.name.replace('.txt', '.jpg')] = dict(
                            bboxs=bboxs,
                            polys=polys,
                            labels=bboxs_labels,
                        )

        self.create_annotation(captured_dataset, dataset_obj, model_type)
        dataset_obj.labels.set(labels)
        dataset_obj.annotated_frames.set(annotated_frames)
        return Response({"data_failed_to_save": data_failed_to_save})

    

class DeploymentHistoryViewSet(ModelViewSet):
    serializer_class = DeploymentHistorySerializer
    # permission_classes = (GenericObjectPermissions,)
    queryset = DeploymentHistory.objects.all()
    perms_map = {
        'GET': ['annotation.view_deploymenthistory'],
        'POST': ['annotation.add_deploymenthistory'],
        'PATCH': ['annotation.change_deploymenthistory'],
        'PUT': ['annotation.change_deploymenthistory'],
        'DELETE': ['annotation.delete_deploymenthistory'],
    }

    def get_queryset(self):
        queryset = self.queryset

        model = self.request.GET.get("model")
        if model:
            queryset = queryset.filter(model_id=model)
        return queryset



class ModelConnectionViewSet(ModelViewSet):
    serializer_class = ModelConnectionSerializer
    # permission_classes = (GenericObjectPermissions,)
    queryset = ModelConnection.objects.all()
    perms_map = {
        'GET': ['panel.view_modelconnection'],
        'POST': ['panel.add_modelconnection'],
        'PATCH': ['panel.change_modelconnection'],
        'PUT': ['panel.change_modelconnection'],
        'DELETE': ['panel.delete_modelconnection'],
    }

    def get_queryset(self):
        os_type = self.request.GET.getlist("type")
        queryset = self.queryset
        if os_type:
            queryset = queryset.filter(compute__in=os_type)
        return queryset

    def paginate_queryset(self, queryset):
        if self.request.query_params.get('no_page'):
            return None
        return super().paginate_queryset(queryset)

