import os
import sys
import time
from datetime import timedelta

sys.path.append(os.environ["BASE_DIR"])
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jarvis_jail.settings")
application = get_wsgi_application()

from django.utils import timezone

from annotation.models import (
    DeploymentHistory,
)

class Train:

    def process(self, *args, **kwargs):
        deployments = DeploymentHistory.objects.filter(status="training")
        for deployment in deployments:
            if deployment.total_data:
                if deployment.created + timedelta(seconds=deployment.total_data * float(deployment.model.time_per_frame)) <= timezone.now():
                    deployment.status = "trained"
                    deployment.save()
            else:
                deployment.status = "failed"
                deployment.save()


    def start(self):
        while True:
            self.process()

if __name__ == "__main__":
    Train().start()
    time.sleep(10)

