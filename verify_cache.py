import json
d = json.load(open("articles_cache.json", encoding="utf-8"))
print("Total articles:", len(d))
for a in d[:5]:
    print(f"  [{a['index']:02d}]", a["label"])
