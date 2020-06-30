import requests
import json
pload = json.dumps([{"id":1,"content":"This is not a logistics"}])
r = requests.post("http://127.0.0.1:9891/infer_content",data = pload)
print(json.dumps(json.loads(r.text),indent=4))
