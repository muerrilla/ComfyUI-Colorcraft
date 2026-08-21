import json, sys, urllib.request

today, today_ts, gist_id, token = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

clones_data = json.load(open('/tmp/clones.json'))
views_data  = json.load(open('/tmp/views.json'))
gist_data   = json.load(open('/tmp/gist.json'))

clones_row      = next((d for d in clones_data.get('clones', []) if d['timestamp'] == today_ts), {})
views_row       = next((d for d in views_data.get('views',   []) if d['timestamp'] == today_ts), {})

clones          = clones_row.get('count',   0)
unique_cloners  = clones_row.get('uniques', 0)
views           = views_row.get('count',    0)
unique_visitors = views_row.get('uniques',  0)

print(f"Today: {today} | clones={clones} unique_cloners={unique_cloners} views={views} unique_visitors={unique_visitors}")

filename = list(gist_data['files'].keys())[0]
current  = gist_data['files'][filename]['content'].rstrip('\n')

if any(line.startswith(today + ',') for line in current.splitlines()):
    print(f"Row for {today} already exists — skipping.")
    sys.exit(0)

new_content = current + f"\n{today},{clones},{unique_cloners},{views},{unique_visitors}"

payload = json.dumps({'files': {filename: {'content': new_content}}}).encode()
req = urllib.request.Request(
    f"https://api.github.com/gists/{gist_id}",
    data=payload,
    method='PATCH',
    headers={
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json',
    }
)
urllib.request.urlopen(req)
print("Gist updated successfully.")
