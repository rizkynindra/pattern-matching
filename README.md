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

# Running performance test in locust

locust -f locustfile.py --host=http://pattern-matching-api-pattern-matching.apps.ocp-drc.bpjsketenagakerjaan.go.id


# Running unit test

python -m unittest -v unit_test.py

