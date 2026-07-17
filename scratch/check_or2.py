import urllib.request, json
url = 'https://openrouter.ai/api/v1/models'
req = urllib.request.urlopen(url)
data = json.loads(req.read())['data']
for m in data:
    b = m.get('benchmarks')
    if b and 'artificial_analysis' in b:
        print(f"{m['id']} -> {b['artificial_analysis']}")
