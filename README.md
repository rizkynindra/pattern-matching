# pattern-matching
general
1. pastikan pakai python versi 3.10.12
2. source venv/bin/activate
3. python main.py

imgSimilarity
1. pastikan template dan dataset sudah dipasang dan 
2. python imgSimilarity_service.py

emailSimilarity
2. python emailSimilarity.py

phoneValidation
2. python phoneValidation


# deploy ke openshift

docker build --platform=linux/amd64 -t pattern-matching-api:v1.1.1 .

docker login -u pti-dev -p $(oc whoami -t) default-route-openshift-image-registry.apps.ocp-drc.bpjsketenagakerjaan.go.id

docker tag pattern-matching-api:v1.1.1 default-route-openshift-image-registry.apps.ocp-drc.bpjsketenagakerjaan.go.id/pattern-matching/pattern-matching-api:v1.1.1

docker push default-route-openshift-image-registry.apps.ocp-drc.bpjsketenagakerjaan.go.id/pattern-matching/pattern-matching-api:v1.1.1

# Running performance test in locust

locust -f locustfile.py --host=http://pattern-matching-api-pattern-matching.apps.ocp-drc.bpjsketenagakerjaan.go.id


# deploy ke openshift devsecops

docker build --platform=linux/amd64 -t pattern-matching-dev:v1.1.0 .

docker login -u pti-dev -p $(oc whoami -t) default-route-openshift-image-registry.apps.ocp-drc.bpjsketenagakerjaan.go.id

docker tag pattern-matching-api:v1.1.0 default-route-openshift-image-registry.apps.ocp-drc.bpjsketenagakerjaan.go.id/pattern-matching/pattern-matching-api:v1.1.0

docker push default-route-openshift-image-registry.apps.ocp-drc.bpjsketenagakerjaan.go.id/pattern-matching/pattern-matching-api:v1.1.0

# Running performance test in locust

locust -f locustfile.py --host=http://pattern-matching-api-pattern-matching.apps.ocp-drc.bpjsketenagakerjaan.go.id


# Running unit test

python -m unittest -v unit_test.py

