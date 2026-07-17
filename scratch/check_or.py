import urllib.request, json
url = 'https://openrouter.ai/api/v1/models'
req = urllib.request.urlopen(url)
data = json.loads(req.read())['data']
for m in data[:10]:
    print(f"{m['id']} Benchmarks: {m.get('benchmarks')}")
