import os
from dotenv import load_dotenv
import requests

load_dotenv()

client_id = os.getenv("MAPPLS_CLIENT_ID")
client_secret = os.getenv("MAPPLS_CLIENT_SECRET")

resp = requests.post(
    "https://outpost.mappls.com/api/security/oauth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    },
)
token = resp.json().get("access_token")

resp2 = requests.get(
    "https://atlas.mappls.com/api/places/search/json",
    params={"query": "Cubbon Park", "region": "ind"},
    headers={"Authorization": f"Bearer {token}"},
)
print("AUTOSUGGEST STATUS:", resp2.status_code)
print("AUTOSUGGEST BODY:", resp2.text)